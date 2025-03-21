import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath('_given'))

from utils import part4Plots


with open('part4/results/result_MLP1.json', 'r') as file:
    results_MLP1 = json.load(file)
with open('part4/results/result_MLP2.json', 'r') as file:
    results_MLP2 = json.load(file)
with open('part4/results/result_CNN3.json', 'r') as file:
    results_CNN3 = json.load(file)
with open('part4/results/result_CNN4.json', 'r') as file:
    results_CNN4 = json.load(file)
with open('part4/results/result_CNN5.json', 'r') as file:
    results_CNN5 = json.load(file)


part4Plots([results_MLP1])
part4Plots([results_MLP2])
part4Plots([results_CNN3])
part4Plots([results_CNN4])
part4Plots([results_CNN5])
