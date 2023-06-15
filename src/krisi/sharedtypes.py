from krisi.utils.enums import ParsableEnum


class Task(ParsableEnum):
    regression = "regression"
    classification = "classification"
    multi_classification = "multi_classification"
