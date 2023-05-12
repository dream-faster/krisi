def handle_test():
    import os

    if "PYTEST_CURRENT_TEST" in os.environ:
        import matplotlib.pyplot as plt

        plt.close("all")
