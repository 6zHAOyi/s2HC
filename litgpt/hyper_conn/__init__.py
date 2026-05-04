from .hyper_connections import (
    get_init_and_expand_reduce_stream_functions
)

# export with mc prefix, as well as mHC

from .mhc import (
    get_expand_reduce_stream_functions as mc_get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions as mc_get_init_and_expand_reduce_stream_functions
)

from .mhc_lite import (
    get_expand_reduce_stream_functions as mhclite_get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions as mhclite_get_init_and_expand_reduce_stream_functions
)

from .shc import (
    get_expand_reduce_stream_functions as shc_get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions as shc_get_init_and_expand_reduce_stream_functions
)

flag = False

def hyper_conn_init_func(hyper_conn_type: str, hyper_conn_n: int):
    global flag
    if not flag:
        print(f"HYPER_CONN: USING {hyper_conn_type} with {hyper_conn_n} streams")
        flag = True

    if hyper_conn_type == "none":
        return get_init_and_expand_reduce_stream_functions(hyper_conn_n, disable = True)
    elif hyper_conn_type == "hc":
        return get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc":
        return mc_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc_lite":
        return mhclite_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "shc":
        return shc_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "kromhc":
        from .kromhc import get_init_and_expand_reduce_stream_functions as kromhc_get_init_and_expand_reduce_stream_functions
        return kromhc_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    else:
        raise ValueError(f"Invalid hyper connection type: {hyper_conn_type}")

