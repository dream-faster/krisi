import os


def handle_test():
    if "PYTEST_CURRENT_TEST" in os.environ:
        import matplotlib.pyplot as plt

        plt.close("all")


def is_testing() -> bool:
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    else:
        return False
