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


# def is_launched_from_krisi(library_name: str = "krisi") -> bool:
#     # Get the traceback information
#     frames = inspect.stack()

#     # Iterate over the frames in the traceback
#     for i, frame_info in enumerate(frames[1:]):
#         try:
#             module = inspect.getmodule(frame_info[0])
#             module_name = module.__name__ if hasattr(module, "__name__") else None
#         except:
#             module_name = None
#         if module_name == library_name:
#             return True

#     # If no parent library found in the traceback, assume called from another parent
#     return False
