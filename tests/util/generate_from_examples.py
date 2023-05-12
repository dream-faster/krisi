import os

file_content = ["# flake8: noqa"]

for entry in os.scandir("docs/examples"):
    if entry.is_file():
        file_name_split = entry.name.split(".")
        if file_name_split[-1] == "py":
            file_content.append(
                f"def test_{file_name_split[0]}():\n\tfrom docs.examples import {file_name_split[0]} # noqa"
            )

    with open("tests/test_examples.py", "w") as f:
        f.write("\n\n".join(file_content))
