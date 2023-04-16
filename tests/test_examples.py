def test_all_examples():
    import os

    for entry in os.scandir("examples"):
        if entry.is_file():
            if entry.name.split(".")[-1] == "py":
                print("\n\n\n============================")
                print(f"--- Running {entry.name} ---")
                print("============================")
                string = f"from examples import {entry.name}"[:-3]
                exec(string)
