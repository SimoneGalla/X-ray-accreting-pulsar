from PulsarListComprehension import dataGenerator
from Compare import storeVariables
import os

""" This code generates the simulated data of a chosen length, then saves them to a parquet file.

 The last functions reads them back and tries a fit where stores the fitting variables.
 
 The data are generated with List comprehension, if you want or need a better efficiency please use PulsarPool.py
 
 Check the other files to understand better what the functions do. 
 
 This file does everything, from data creation to comparison, at once. 
 I believe one might not need everything, so I recommend checking the other files to get what needed. 
 
 The other files perform the following operations:
 
 -PulsarPool generates the chosen number of datasets and saves them to a chosen file 
 -PulsarListComprehension does the same but slower, with List Comprehension (for loops, the data generation in this file is from here)
 -ReadParquet defines the needed functions to read the database
 -Compare gives an example on how to use the read data fitting a pulse profile (storeVariables in this file is from here)
 -GetPulseProfileFromInput gives back the closest pulse profile associated to the input parameters
 
 """



size_data = 3

file_name = 'bigdata_length_{0}'.format(size_data)

folder_path = '/Users/Utente/Desktop'

file_path = os.path.join(folder_path, file_name)  # One can also directly write a file that wants to use here

dataGenerator(size_data, file_path) #Generates a list of dictionaries containing all the simulated data, check NoPlotParquet for more information


#This reads back the file and fits to an example of experimental data exploiting the fitting parameters
param1_fit, param2_fit, param3_fit, rot_i_fit, rot_a_fit, magnc_fit, shift_fit = storeVariables(file_path)