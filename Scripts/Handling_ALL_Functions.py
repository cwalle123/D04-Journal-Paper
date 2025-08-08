"""This file deals with all of the important functions that are used all throughout the project"""

##############################################################################################################

# External imports
import numpy as np
import pandas as pd
import glob
import os

# Internal imports
import Scripts.Data_ALL_importer

###################################################################################################################################################################################################
"""Functions for purging, saving and loading data"""

_save_path = "Cached data\\"

def _save_table(data_table:pd.DataFrame, short_name:str)-> None:
    '''This function saves a pandas dataframe as
        a .pkl, it will be saved with the short name, 
        use that to access it'''
    
    data_table.to_pickle(_save_path + short_name + ".pkl")
    # note! this does not save headers or indexes. might need to change that depending on how we do
    return

def _load_table(short_name:str)->pd.DataFrame:
    '''This function reads a pkl and turns it into 
        a panda Dataframe. access it with the same name 
        used in the save_csv() function if file doesn't exist it returns none'''
    
    try:
        return pd.read_pickle(_save_path + short_name + ".pkl")
    except FileNotFoundError:
        return None

def export_to_csv(data_table:pd.DataFrame, name:str)-> None:
    '''This function exports the table to CSV.
        Note! if you want to save your progress, use the save_table() function instead,
        as the CSV is not reversibly saved (metadata is lost)'''
    
    data_table.to_csv(_save_path + name)
    return None

def purge_cache(confirmation:bool = False)->None:
    '''gets rid of all the files in the cache.\n
    Warning! don't do this unless you're sure you want to\n
    you will have to generate all the data again'''

    print("purging cache...")
    if confirmation != True:
        raise PermissionError("You did not give confirmation for purging the cache. are you sure you want to do this?")

    files = glob.glob(_save_path + "*")
    for file in files:
        if os.path.isfile(file):
            os.remove(file)
    print("cache purged")

def create_cache()->None:
    """runs through all the datatypes and generates the cache, will take some time"""
    print("Let there be cache...")
    codes = ["LT","LLS_A","LLS_B","CAM"]
    tows = range(1,32)

    for code in codes:
        for tow in tows:
            get_processed_data(tow,code, True)
    print("Cache created!")

def create_processed_cache()->None:
    '''generates the cache for the time and space synced tows'''
    print("Let there be cache...")
    tows = range(1,32)
    for tow in tows:
        get_synced_data(tow, spacesynced=False, overwrite=True)
        get_synced_data(tow, spacesynced=True, overwrite=True)
    print("Cache created!")

################################################################################################################
"""Functions for calling data"""

def get_synced_data(tow:int, sensor_type:str, overwrite=False, helper=False)->pd.DataFrame:
    '''
    This function handles ALL the grabbing and processing of the raw data\n
    call this and it will do all the stuff for you, no other functions needed\n
    the function parameters are:\n
    \n
    tow:int, the index of the tow from 1 to 32\n
    sensor_type:str, the type of data to get. valid keys are: "LT","LLS_A","LLS_B","CAM"\n
    overwrite:bool (optional), If this is true, the function will ignore the cache\n
    and reprocess the raw data. False by default. only do this if something in the processing\n
    has changed, or if the raw data has changed.\n
    helper is a variable that should always be false when using (it's just to make it work with get synced data)\n
    (it circumvents the messages and the saving process since sync will save instead)
    '''

    # generate consistent name:
    # first check if key is valid
    if sensor_type not in ["LT","LLS_A","LLS_B","CAM"]:
        raise KeyError(f"the Key {sensor_type} was invalid: No such data exists")
    # then that tow exists:
    if tow not in range(1,32):
        raise IndexError(f"Tow ID {tow} is out of range")
    # set the name
    name = sensor_type + "_" + str(tow)
    if not helper: # ignore all the data
        # check if file exists:
        data = _load_table(name)

        if data is not None and not overwrite:
            #if true the data already exists, return it:
            return data
        # else the data doesn't exist, grab it
        print(f"No file with code {name} cached. Generating new data...")
    match sensor_type:
        case "LT":
            # Laser Tracker
            data = np.array(LT_exceltolist()[tow-1]).T
            processesed_data = _handle_LT(*data[1:], tow)

        case "CAM":
            # Camera Data
            data = np.array(CAM_exceltolist()[tow-1]).T
            processesed_data = _handle_camera(*data[:4])

        case "LLS_A":
            # Laser Line Sensor 1
            data = np.array(LLS_exceltoarray()[tow*2-2]).T
            processesed_data = _handle_LLS(*data[:4])

        case "LLS_B":
            # Laser Line Sensor 2
            data = np.array(LLS_exceltoarray()[tow*2-1]).T
            processesed_data = _handle_LLS(*data[:4])
    if not helper:
        _save_table(processesed_data, name) # save the data
    return processesed_data

def main():
    "Hi"

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
