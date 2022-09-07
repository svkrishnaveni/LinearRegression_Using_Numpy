#!/usr/bin/env python
'''
This script contains code for applying linear regression model (trained using training data) to test data and estimating MSE

Author: Sai Venkata Krishnaveni Devarakonda
Date: 03/15/2022
'''

import matplotlib.pyplot as plt
import numpy as np
from utilities import fit_lin_reg,predict,mse,Load_data

str_path_1b = './1b_training_data.txt'
str_path_1c = './1c_test_data.txt'
for i in range(7):
    arr_trainfeatures_set, train_t, x = Load_data(str_path_1b, i, 6)
    arr_testfeatures_set, test_t, x_t = Load_data(str_path_1c, i, 6)

    coef = fit_lin_reg(arr_trainfeatures_set, train_t)
    y_pred_test = predict(arr_testfeatures_set, coef)
    y_pred_train = predict(arr_trainfeatures_set, coef)

    mse_test = mse(y_pred_test, test_t)
    mse_train = mse(y_pred_train, train_t)
    print('for d=' + str(i) + ' train MSE=' + str(mse_train))
    print('for d=' + str(i) + ' test MSE =' + str(mse_test))
    print('\n')

    x_sorted = np.argsort(x_t)
    plt.plot(x_t[x_sorted], test_t[x_sorted], '*g')
    # plotting given features with predicted targets
    plt.plot(x_t[x_sorted], y_pred_test[x_sorted], '-b')
    # giving labels to x and y axis
    plt.xlabel('x test')
    plt.ylabel('signal amplitude')
    plt.legend(['original signal', 'bestfit'])
    # giving title to the plotted graph
    plt.title('discrete signal test points and best fit with function depth = ' + str(i))
    plt.savefig('./plots/testdata_Bestfit_d_' + str(i) + '.png', dpi=150)
    plt.show()