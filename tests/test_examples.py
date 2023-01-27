def test_all_examples():
    import os

    for entry in os.scandir("examples"):
        if entry.is_file():
            string = f"from examples import {entry.name}"[:-3]
            exec(string)
