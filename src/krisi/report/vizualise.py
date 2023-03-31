import pandas as pd


def plot_y_and_predictions(y: pd.Series, preds: pd.DataFrame) -> None:
    df = pd.DataFrame({"y": y, "predictions": preds.squeeze()})
    df.plot(mark_right=False, figsize=(20, 5), grid=True, alpha=0.7)
