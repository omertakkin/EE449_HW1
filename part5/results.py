import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath('_given'))

from utils import part5Plots


with open('part5/results/result_CNN5.json', 'r') as file:
    results_CNN5 = json.load(file)




part5Plots([results_CNN5])


