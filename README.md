## Table of Contents

1. [Code preparation](#code-preparation)
2. [Environment preparation](#environment-preparation)
3. [Dataset preparation](#dataset-preparation)
4. [Model preparation](#model-preparation)
5. [Configuration preparation](#configuration-preparation)
6. [Launch Pretraining](#launch-pretraining)
7. [Personalize](#personalize)
8. [Acknowledgements](#acknowledgements)

## Code preparation
Clone this repository

## Environment preparation
1. To install all required dependencies before pretraining, first run:
```
pip install "litgpt[all]"
pip install einops
```

## Dataset preparation
Download the finewebedu dataset under `/fineweb_edu`.


## Model preparation
1. Download the tokenizer:
```
litgpt download Qwen/Qwen3-0.6B \
  --tokenizer_only true
```
Then the tokenizer will be saved under `/checkpoints`.

## Configuration preparation
Check the `config_hub/pretrain/qwen3-0.6b.yaml` and adjust your settings accordingly.

## Launch pretraining
Run the following command to launch training on Qwen3:
```
litgpt pretrain --config config_hub/pretrain/qwen3-0.6b.yaml \
        --hyper_conn_type "shc" \
        --hyper_conn_n 4
```
You can change the hyper-connection method with the option `--hyper_conn_type` and available methods are:
- Spectral-Sphere-Constrained Hyper-Connections (sHC)    -> `shc`
- Manifold-Constrained Hyper-Connections (mHC)    -> `mhc`
- Manifold-Constrained Hyper-Connections with Permutations (mHC-lite)    -> `mhc_lite`
- Manifold-Constrained Hyper-Connections with Kronecker-Product (KromHC)    -> `kromhc`
- Hyper-Connections    -> `hc`
- Default Residual Connection    -> `none`

You can also change the number of residual streams with the option `--hyper_conn_n`.

## Personalize
Our project is built on the [`LitGPT`](https://github.com/Lightning-AI/litgpt) framework, it is highly flexible and extensible. 

If you want to train on different datasets or other base model architectures, you can easily do so by leveraging LitGPT's built-in support. 

1. **Check Supported Models/Datasets:** Refer to the official [LitGPT documentation](https://github.com/Lightning-AI/litgpt) for a complete list of supported models and data preparation steps.
2. **Modify the Configuration:** Once your new data or model is ready, simply update the corresponding paths and training settings in your `.yaml` configuration file (e.g., `config_hub/pretrain/qwen3-0.6b.yaml`).

All our hyper-connection methods (e.g., `shc`, `mhc`) are designed as plug-and-play modules and are fully compatible with LitGPT's standard training pipeline.

## Acknowledgements

Our implementation stands on the shoulders of several excellent open-source projects. We would like to express our gratitude to:

- [LitGPT](https://github.com/Lightning-AI/litgpt): For providing a robust and scalable framework for large language model pretraining.
- [mHC-lite](https://github.com/FFTYYY/mhc-lite): For the implementations of Manifold-Constrained Hyper-Connections (`mhc`) and their own Permutation-based Hyper-Connections(`mhc-lite`).
- [KromHC](https://github.com/lucidrains/kromhc): For the implementation of Kronecker-Product based Hyper-Connections (`kromHC`).
- [HC](https://github.com/lucidrains/hyper-connections): For the implementation of Hyper-Connections (`HC`).