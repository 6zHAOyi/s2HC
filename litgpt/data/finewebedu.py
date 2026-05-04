import os
import glob
import random
import torch
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader
from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer
from litdata.streaming import TokensLoader
import pyarrow.parquet as pq

def tokenize_fn(filepath: str, tokenizer: Tokenizer):
    parquet_file = pq.ParquetFile(filepath)
    
    for batch in parquet_file.iter_batches(batch_size=50000, columns=["text"]):
        
        texts = batch.column("text").to_pylist()
        for text in texts:
            tokens = tokenizer.encode(str(text), eos=True)
            yield tokens.to(torch.int32)


@dataclass
class FineWebEdu(DataModule):
    data_path: str | Path = Path("fineweb_edu_bin")
    parquet_dir: str = "fineweb_edu" 
    val_split_fraction: float = 0.005
    seed: int = 42
    num_workers: int = 2

    tokenizer: Tokenizer | None = field(default=None, repr=False, init=False)
    batch_size: int = field(default=1, repr=False, init=False)
    seq_length: int = field(default=2048, repr=False, init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.data_path_train = str(self.data_path).rstrip("/") + "/train"
        self.data_path_val = str(self.data_path).rstrip("/") + "/val"

    def connect(self, tokenizer: Tokenizer | None = None, batch_size: int = 1, max_seq_length: int | None = 2048) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1

    def prepare_data(self) -> None:
        from litdata import optimize

        if Path(self.data_path_train).is_dir() and Path(self.data_path_val).is_dir():
            print(f"[*] Data prepared: {self.data_path}. Skip")
            return

        pq_files = glob.glob(os.path.join(self.parquet_dir, "*.parquet"))
        assert len(pq_files) > 0, f"can't find any parquet file in {self.parquet_dir}!"
        
        pq_files = sorted(pq_files)
        random.seed(self.seed)
        random.shuffle(pq_files)
        
        val_count = max(1, int(len(pq_files) * self.val_split_fraction))
        val_files = pq_files[:val_count]
        train_files = pq_files[val_count:]
        
        print(f"[*] {len(train_files)} Train splits, {len(val_files)} splits.")

        item_loader = TokensLoader(block_size=self.seq_length)

        print("[*] Generating Train Data...")
        optimize(
            fn=partial(tokenize_fn, tokenizer=self.tokenizer),
            inputs=train_files,
            output_dir=self.data_path_train,
            num_workers=min(32, (os.cpu_count() or 2) - 1),
            chunk_bytes="200MB",
            item_loader=item_loader
        )
        
        print("[*] Generating Val Data")
        optimize(
            fn=partial(tokenize_fn, tokenizer=self.tokenizer),
            inputs=val_files,
            output_dir=self.data_path_val,
            num_workers=min(8, (os.cpu_count() or 2) - 1),
            chunk_bytes="200MB",
            item_loader=item_loader
        )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset
        train_dataset = StreamingDataset(
            input_dir=self.data_path_train,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
        )
        return StreamingDataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset
        val_dataset = StreamingDataset(
            input_dir=self.data_path_val,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
        )
        return StreamingDataLoader(val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True)