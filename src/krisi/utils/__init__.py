from importlib.util import find_spec

from krisi.utils.devutils.environment_checks import is_testing
from krisi.utils.state import GlobalState, RunType, set_global_state

if is_testing():
    set_global_state(GlobalState(run_type=RunType.test))
else:
    set_global_state(GlobalState(run_type=RunType.dev))
