#!/usr/bin/env python
'''
This script contains code for applying linear regression model trained using training data of first 20 observations to test data and estimating MSE

Author: Sai Venkata Krishnaveni Devarakonda
Date: 03/15/2022
'''
import matplotlib.pyplot as plt
import numpy as np
from utilities import fit_lin_reg,predict,mse,Load_data

str_path_1b_traindata = './1b_training_data.txt'
str_path_1c_testdata = '1c_test_data.txt'
for d in range(7):
    arr_trainfeatures_set, train_t, x = Load_data(str_path_1b_traindata, d, 6)
    # select only first 20 rows of train data
    arr_trainfeatures_set_tmp, train_t_tmp, x_tmp = arr_trainfeatures_set[:21, :], train_t[:21, ], x[:21, ]
    arr_testfeatures_set, test_t, y = Load_data(str_path_1c_testdata, d, 6)
    coef = fit_lin_reg(arr_trainfeatures_set_tmp, train_t_tmp)
    y_pred_test = predict(arr_testfeatures_set, coef)
    y_pred_train = predict(arr_trainfeatures_set_tmp, coef)

    # get mse for both train_predictions and test_predictions
    mse_test = mse(y_pred_test, test_t)
    mse_train = mse(y_pred_train, train_t_tmp)
    print('for d=' + str(d) + ' train error=' + str(mse_train))
    print('for d=' + str(d) + ' test error =' + str(mse_test))
    print('\n')

    x_sorted = np.argsort(y)
    plt.plot(y[x_sorted], test_t[x_sorted], '*g')
    # plotting given features with predicted targets
    plt.plot(y[x_sorted], y_pred_test[x_sorted], '-b')
    # giving labels to x and y axis
    plt.xlabel('x test')
    plt.ylabel('signal amplitude')
    plt.legend(['original signal', 'bestfit'])
    # giving title to the plotted graph
    plt.title('Linear regression training(20) with function depth d = ' + str(d))
    plt.savefig('./plots/linreg_first20_testdata_Bestfit_d_' + str(d) + '.png', dpi=150)
    plt.show()

