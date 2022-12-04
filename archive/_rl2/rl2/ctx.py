from contextvars import ContextVar
import os
from pathlib import Path
from contextvars import ContextVar

default_config = {
    # if default log_dir changes in cli.py, then change here, too
    'log_dir': os.path.join(Path.home(), 'rl2-runs')
}
ctx_dict = ContextVar('config', default=default_config)

ctx_device = ContextVar(
    'device',
    default='cpu'
)

ctx_seed = ContextVar(
    'seed',
    default=42
)

ctx_debug_flag = ContextVar(
    'debug_flag',
    default=False
)

ctx_debug_flag = ContextVar(
    'debug_flag',
    default=False
)