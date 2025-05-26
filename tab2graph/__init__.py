from . import logger
from . import time_budget

def _check_dgl_version():
    import dgl
    required_version = "2.1a240205"
    parts = dgl.__version__.split('+')
    current_version = parts[0]
    if current_version != required_version:
        raise RuntimeError(
            f"Required DGL version {required_version} but the installed version is {current_version}."
        )

_check_dgl_version()
