import pkgutil
from typing import List, Optional, Union

import pandas as pd


def plot_y_predictions(
    y: pd.Series,
    preds: Union[List[pd.Series], pd.DataFrame],
    title: Optional[str] = "",
    index_name: str = "index",
    value_name: str = "value",
    variable_name: str = "models",
) -> None:
    if isinstance(preds, List):
        preds = pd.concat(preds, axis="columns")
    df = y.to_frame()
    df = pd.concat([y, preds], axis="columns")

    labels = dict(index=index_name, value=value_name, variable=variable_name)

    if pkgutil.find_loader("plotly"):
        fig = df.plot(
            backend="plotly",
            title=title,
            labels=labels,
        )
        for column in df.columns:
            if column != "y":
                fig.update_traces(selector=dict(name=column), opacity=0.5)
        fig.show()
    elif pkgutil.find_loader("matplotlib"):
        df.plot(
            backend="matplotlib",
            mark_right=False,
            figsize=(20, 5),
            grid=True,
            alpha=0.7,
            xlabel=labels["index"],
            ylabel=labels["value"],
            title=title,
        )
    else:
        raise AssertionError(
            "Neither `Plotly` or `Matplotlib` is installed. `Ploty` is preferred over `Matplotlib`."
        )
