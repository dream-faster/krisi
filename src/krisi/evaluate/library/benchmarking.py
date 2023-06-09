from random import choices
from typing import List, Optional, Union

import numpy as np
import pandas as pd


class RandomClassifier:
    def __init__(
        self, all_classes: List[int], probability_mean: Optional[List[float]] = None
    ) -> None:
        self.all_classes = all_classes
        if probability_mean is not None:
            assert len(probability_mean) == len(all_classes)
        self.probability_mean = (
            [1 / len(all_classes) for _ in range(len(all_classes))]
            if probability_mean is None
            else probability_mean
        )

    def predict(self, X: pd.Series) -> Union[pd.Series, pd.DataFrame]:
        predictions = pd.Series(
            choices(population=self.all_classes, k=len(X)),
            index=X.index,
            name="predictions_RandomClassifier",
        )
        probabilities = pd.concat(
            [
                pd.Series(
                    np.random.normal(prob_mean, 0.1, len(X)).clip(0, 1),
                    index=X.index,
                    name=f"probabilities_RandomClassifier_{associated_class}",
                )
                for associated_class, prob_mean in zip(
                    self.all_classes, self.probability_mean
                )
            ],
            axis="columns",
        )
        probabilities = probabilities.div(probabilities.sum(axis=1), axis=0)

        return pd.concat([predictions, probabilities], axis="columns")
