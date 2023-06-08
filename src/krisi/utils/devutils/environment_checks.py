import os
import sys


def handle_test():
    if "PYTEST_CURRENT_TEST" in os.environ:
        import matplotlib.pyplot as plt

        plt.close("all")


def is_testing() -> bool:
    in_testing = False
    if "PYTEST_CURRENT_TEST" in os.environ:
        in_testing = True

    script_name = os.path.basename(sys.argv[0])
    if script_name in ["pytest", "py.test"]:
        in_testing = True

    return in_testing
