#!/usr/bin/env python
'''
This script contains code for applying linear regression to fit the signal using periodic basis functions
for different function depths 'd' also plots the resulting functions along with original data

Author: Sai Venkata Krishnaveni Devarakonda
Date: 03/15/2022
'''

from utilities import plot_1b

str_path_1b_traindata = './1b_training_data.txt'
plot_1b(str_path_1b_traindata,6,6, save_figures=False)