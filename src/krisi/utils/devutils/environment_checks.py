import os
import sys


def handle_test() -> None:
    if is_testing():
        import matplotlib.pyplot as plt

        plt.close("all")


def is_testing() -> bool:
    in_testing = False
    if "PYTEST_CURRENT_TEST" in os.environ:
        in_testing = True

    for argv_ in sys.argv:
        script_name = os.path.basename(argv_)
        if script_name in ["pytest", "py.test"]:
            in_testing = True

    return in_testing
