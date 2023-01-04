
from typing import List

from krisi.evaluate.scorecard import ScoreCard


def compare(scorecards:List[ScoreCard], sort_metric_key:str="rmse")->None:
    
    print(f"{sort_metric_key:<30s}")
    scorecards.sort(reverse=True, key=lambda x: x[sort_metric_key].result)
    
    for scorecard in scorecards:
        print(f"{scorecard.model_name:>30s} {scorecard[sort_metric_key].result:<15.5}")
    
    