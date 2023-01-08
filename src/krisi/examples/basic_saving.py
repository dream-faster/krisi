import numpy as np

from krisi import eval
from krisi.evaluate.type import SaveModes

eval(y=np.random.rand(1000), predictions=np.random.rand(1000)).save(
    save_modes=[
        SaveModes.minimal,
        SaveModes.obj,
        SaveModes.text,
        SaveModes.svg,
        SaveModes.html,
        SaveModes.text,
    ]  # Optional, defaults to [ SaveModes.minimal, SaveModes.obj, SaveModes.text ]
)
