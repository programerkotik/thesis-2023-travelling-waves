import sys
import os
from pathlib import Path

file_path = __file__
project_path = Path(file_path).parent.parent.parent

print(file_path)
print(project_path)

print(os.getcwd())

# os.chdir("../..")

sys.path.append("../..")

import numpy as np


np.loadtxt(f"{project_path}/res/16-11-2022-res/x.txt")


from src import *