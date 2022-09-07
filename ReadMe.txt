Requirements and dependencies:
    1. Numpy
    2. re  (regular expression)
    3. Math
    4.matplotlib.pyplot
    5.os
    6.random

Loading train and test data:

I have copied the data from web generated php files into text files (1b_training_data.txt,1c_test_data.txt, data_2c2d3c3d_program.txt, data_test_2a3a, data_train_2a3a)
I use custom functions (Load_data, Load_data_2, Load_data_logisticreg_train, Load_data_logisticreg_test) to load and convert .txt files to numpy arrays
Please place input .txt files (1b_training_data.txt,1c_test_data.txt, data_2c2d3c3d_program.txt)  in the current working directory

Running instructions:

run the following commands:
	1)for applying regression learner to the data set and plotting the resulting function for ”function depth” 0, 1, 2, 3, 4, 5, and 6. :	run_1b.py
	2)for evaluating regression functions by computing the error on the test data points that were generatedand for comparing the error results :	run_1c.py
	3)for evaluating of part 1) and 2) using only the first 20 elements of the training data set 	:	run_1d.py
	


Inorder to use custom data, copy the data to a text file and use Load_data* functions with appropriate paths to .txt files.

All supporting functions are mainly located in utilities.py
