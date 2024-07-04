import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from ReadParquet import printKeys, readParquetToDict

""" The idea behind this file is to try to read the simulated data from a file,
with functions that return a list of dictionaries with the data in, and then compare each line with
the experimental data, in order to find the simulated set of parameters that best suits the experimental pattern.

The main goal of this file is NOT to be used in future data analysis, even if it could be helpful, 
but to show how one can read and use the data stored inside the database 

Before running the file please change the file path to the one where the file is stored!
"""

def normalise_weights(w1, w2):

    wsum = w1 + w2
    w1 = w1 / wsum
    w2 = w2 / wsum
    wsum = wsum / np.mean(wsum)
    return w1, w2, wsum


def decompositions():
    """This function creates the decomposition of the total pulse profile into two different pulse profiles coming
    from the different poles from the experimental data

    Returns:
    d1, d2: The two experimental curves that define the decomposition in the two different contributing poles (numpy arrays)"""

    d1 = np.array([2.089433657753390321e+02,2.194589299387001233e+02,2.242701956130024143e+02,2.215828113982570358e+02,2.119044080603044904e+02,1.896290164200222534e+02,1.699201994046265440e+02,1.256523129323831256e+02,8.503688265750899689e+01,7.872871603812254193e+01,6.991923027381139377e+01,5.683178911921619658e+01,3.124156755276437281e+01,7.269036823844857054e+01,6.021225567409459956e+01,3.283549153829007122e-05,2.070746335489954859e+01,2.039204294346190522e+02,4.840049429063090543e+02,7.172415178717548088e+02,8.874827811973464122e+02,9.764660867645188773e+02,9.541305245679710652e+02,8.215813339957139760e+02,6.190850039923557233e+02,4.598887253861233262e+02,3.712954485901983048e+02,3.156898180851563325e+02,2.666241611172789021e+02,2.256346741358247527e+02,2.103769883639451166e+02,2.078976085285444526e+02])
    d2 = np.array([1.706203983036242562e+02,1.627295767828694579e+02,1.508354117504467524e+02,1.387866932644533904e+02,1.222486136182712357e+02,1.134912451761991292e+02,9.134073071527171805e+01,8.393452247999425708e+01,1.066817197404563586e+02,1.153710780759624868e+02,1.233759737954631532e+02,1.483167515129746903e+02,2.492913136389517490e+02,3.399179640856401079e+02,4.636740226352187051e+02,6.800307289423939210e+02,8.913740432455606424e+02,9.093974220853033330e+02,7.202968616231810302e+02,5.014408728270462916e+02,2.981560101312231836e+02,1.266512076523395081e+02,2.776344418581356877e+01,3.598816179894583911e+00,5.163147899177897671e+01,1.027343217254183969e+02,1.227055202846957513e+02,1.252671880914211329e+02,1.319053895669890153e+02,1.488350485237368162e+02,1.574525502093866010e+02,1.655530999612218181e+02])
    d1, d2, dpp = normalise_weights(d1, d2) #All different normalisations
    d1 *= dpp
    d1 /= np.max(dpp)
    d2 *= dpp
    d2 /= np.max(dpp)
    return d1, d2



def manualFit(data_dict):
    """
    This function compares the simulated values that we read from the file with the experimental one, d1.
    This then finds and returns the best fitting line from the simulation, with all the connected parameters.

    Parameters:
        -------------
        data_dict (list[dict]) :  It is the list of dictionaries containing the data we are interested in

    Returns:
        data_dict: same as input
        ok: index of the best line used to fit the pattern
        d1: pattern of the experimental data

    Notes:
        --------
    Probably not the best fitting method, but it is rather fast and easy to understand
    """

    prova = data_dict
    print("Length = ", len(prova))
    summa = 100000000
    d1, d2 = decompositions()
    for i in range(len(prova)):
        diff = d1 - prova[i]["hotspot pattern"]
        somma = np.sum(np.abs(diff))
        if np.abs(somma) < np.abs(summa):
            summa = somma
            ok = i
    return prova[ok], d1


def plottaTutto(fittingDict, d1):

    """ I don't know why the name of this function is in Italian, but it plots everything.

    Parameters:
        -------------
        fittingDict (dict) : The best fitting line of simulated data compared to the experimental ones
        d1 (list): The experimental data

    Notes:
        -------
    Plots the experimental data and the best fitting simulated data on the same graph,
    writing in a box the parameters of the simulated line.
    """
    x = np.linspace(0, 1, 32)
    plt.plot(x, fittingDict["hotspot pattern"])
    plt.plot(x, d1)
    textstr = '\n'.join((
        r'$param1$: %f' % (fittingDict["param1"]),
        r'$param2$: %f' % (fittingDict["param2"]), r'$param3:%f$' % (fittingDict["param3"]),
        r'rotation inclination: %f' % (fittingDict["rotation inclination"]),
        r'rotation azimuth: %f' % (fittingDict["rotation azimuth"]), r'$shift:%f$' % (fittingDict["shift"])))

    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=8, verticalalignment='top', bbox=dict(facecolor='green', alpha=0.1))
    plt.show()


def storeVariables(filepath):
    """
    Parameters:
        ---
        filepath (str): the path of the file where the simulated data are stored

    Returns:
        ---
        param1(float), param2, param3, rot_i, rot_a, magnc, shift: all the parameters that define the best-fitting simulated hotspot pattern

    Notes:
        -------
    This is the "main()". It runs all the other functions in order to just use the filepath to obtain everything
    else that we need from here.
    """

    loaded_data = readParquetToDict(filepath)

    fit, d1 = manualFit(loaded_data)

    plottaTutto(fit, d1)

    param1 = fit["param1"]
    param2 = fit["param2"]
    param3 = fit["param3"]
    rot_i = fit["rotation inclination"]
    rot_a = fit["rotation azimuth"]
    magnc = fit["magnetic colatitude"]
    shift = fit["shift"]

    print("The parameters that fit the best the data are: ", fit)
    return param1, param2, param3, rot_i, rot_a, magnc, shift

###--Decomment to get an example--##

##########

file_name = f'Database_100000'
folder_path = "Database"
file_path = os.path.join(folder_path, file_name)

param1, param2, param3, rot_i, rot_a, magnc, shift = storeVariables(file_path)
