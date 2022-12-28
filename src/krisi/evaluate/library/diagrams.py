from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from krisi.evaluate.type import MetricResult


def display_time_series(
    data: List[MetricResult], name: str = "", width: Optional[float] = None
) -> go.Figure:
    df = pd.DataFrame(data, columns=[name])
    df["iteration"] = list(range(len(data)))
    fig = px.line(
        df,
        x="iteration",
        y=name,
        width=width,
    )
    return fig
