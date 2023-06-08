from krisi.utils.devutils.environment_checks import is_testing
from krisi.utils.state import GlobalState, RunType, set_global_state

if is_testing():
    import matplotlib.pyplot as plt

    set_global_state(GlobalState(run_type=RunType.test))
    plt.switch_backend("Agg")
else:
    set_global_state(GlobalState(run_type=RunType.dev))
