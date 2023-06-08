from dataclasses import dataclass

from krisi.utils.enums import ParsableEnum


class RunType(ParsableEnum):
    dev = "dev"
    test = "test"
    prod = "prod"


@dataclass
class GlobalState:
    run_type: RunType = RunType.dev
    verbose: bool = True


GLOBAL_STATE = GlobalState(run_type=RunType.dev)


def set_global_state(state: GlobalState):
    global GLOBAL_STATE
    GLOBAL_STATE = state


def get_global_state() -> GlobalState:
    return GLOBAL_STATE
