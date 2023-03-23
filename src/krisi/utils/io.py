import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

from rich.console import Console

from krisi.evaluate.type import MetricCategories, PathConst, SaveModes
from krisi.report.console import get_minimal_summary, get_summary

if TYPE_CHECKING:
    from krisi.evaluate.scorecard import ScoreCard


def save_object(obj: "ScoreCard", path: Path) -> None:
    import dill

    final_path = Path(os.path.join(path, Path("scorecard.pickle")))

    with open(final_path, "wb") as file:
        dill.dump(obj, file)


def save_console(
    obj: "ScoreCard",
    path: Path,
    with_info: bool,
    save_modes: List[Union[SaveModes, str]],
) -> None:
    summary = get_summary(
        obj,
        repr=True,
        categories=[el.value for el in MetricCategories],
        with_info=with_info,
    )

    console = Console(record=True, width=120)
    with console.capture():
        console.print(summary)

    if SaveModes.text in save_modes or SaveModes.text.value in save_modes:
        console.save_text(os.path.join(path, Path("console.txt")), clear=False)
    if SaveModes.html in save_modes or SaveModes.html.value in save_modes:
        console.save_html(os.path.join(path, Path("console.html")), clear=False)
    if SaveModes.svg in save_modes or SaveModes.svg.value in save_modes:
        console.save_svg(
            os.path.join(path, Path("console.svg")),
            title="save_table_svg.py",
            clear=False,
        )


def save_minimal_summary(obj: "ScoreCard", path: Path) -> None:
    text_summary = get_minimal_summary(obj)

    final_path = Path(os.path.join(path, Path("minimal.txt")))

    with open(final_path, "w", encoding="utf-8") as f:
        f.write(text_summary)


def load_scorecards(
    project_name: str, path: Union[str, Path] = PathConst.default_eval_output_path
) -> List["ScoreCard"]:
    import os
    import pickle

    if isinstance(path, str):
        path = Path(path)

    path = Path(os.path.join(path, Path(f"{project_name}")))
    files = os.listdir(path)

    loaded_scorecards = []
    for file in files:
        with open(
            Path(os.path.join(path, Path(f"{file}/scorecard.pickle"))), "rb"
        ) as f:
            loaded_scorecards.append(pickle.load(f))

    return loaded_scorecards
