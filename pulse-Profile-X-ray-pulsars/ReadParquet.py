import pandas as pd
import time
import os

""" This python file has been written to read the data from the pulsar x-ray emission simulation.

 The data will be stored in a dictionary and the entries will be printed for future use """


def readParquetToDict(filepath):
    """This function reads the parquet file where the data are stored and transforms the data in order to have a dictionary

    Parameters:
        -----------
        filepath (string) : the path of the file that we want to read the data from

    Returns:
        ---
        A list of the dictionaries of the data

    Notes:
        --------

        Install pyarrow to use this
        Install and import pandas as pd

        """
    file_path = str(filepath)
    loaded_data = pd.read_parquet(file_path, engine='pyarrow')
    file_size = os.stat(file_path)
    print("file size parent file:", file_size.st_size, "bytes")
    data_dict = loaded_data.to_dict(orient="records")

    printKeys(data_dict)

    return data_dict


def printKeys(data_dict):
    """
    Prints the keys of the dictionary used in the project

    Parameters:
        ----------
        data_dict (list[dict]): it should be read (or obtained) from the simulation data

    Returns:
        ---------
        The keys of the dictionary used

    Notes:
        ----------
    This displays the keys but the data_dict that we have is a list of dictionaries,
    so keep it in mind when reading at next steps
    """

    keys = data_dict[1].keys()
    descriptions = [
        "is the \"azimuthal\" parameter of the beam pattern",
        "is the \"inclination\" parameter of the beam pattern",
        "is the power parameter of the beam pattern",
        "is the pulse profile computed with the parameters",
        "is the angle of inclination of the rotation axis",
        "is the azimuthal angle of the rotation axis",
        "is the magnetic colatitude, describes the angle between the possible hotspots and the rotation axis",
        "is the position of the hotspot around the rotation axis",

    ]
    i = 0
    for key in data_dict[1]:

        print("%s: %s\n" %(key, descriptions[i]))
        i += 1


