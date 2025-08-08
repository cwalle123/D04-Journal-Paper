"""This file deals with all of the important functions that are used all throughout the project"""

##############################################################################################################

# External imports
import numpy as np
import pandas as pd
import glob
import os

# Internal imports
from Scripts.constants import z_ref
from Scripts.Data_ALL_importer import LLS_exceltoarray, CAM_exceltolist, LT_exceltolist

################################################################################################################
"""Functions for Laser Tracker"""

def _handle_LT(time: list,
               x: list,
               y: list,
               z: list,
               tow: int) -> pd.DataFrame:
    """
    Build the LT DataFrame and then drop any points
    where y decreases (i.e. the gantry returning).
    """
    import numpy as np
    import pandas as pd

    # 1) build the raw table exactly as before
    rows = len(time)
    tbl = np.empty((rows, 6))
    error_y, error_z = _error_LT(y, z, tow)
    t0 = time_to_float(time[0])

    for i in range(rows):
        tbl[i,0] = time_to_float(time[i]) - t0
        tbl[i,1] = x[i]
        tbl[i,2] = y[i]
        tbl[i,3] = z[i]
        tbl[i,4] = error_y[i]
        tbl[i,5] = error_z[i]

    df = pd.DataFrame(tbl, columns=[
        "time","x","y","z","error_LT","z error"
    ])

    # 2) KEEP ONLY THE OUTBOUND SWEEP: drop rows where y dips
    #    for the very first row, diff() is NaN → fill with True so we keep it
    # forward_mask = df["y"].diff().fillna(1) > 0
    # df = df[forward_mask].reset_index(drop=True)

    return df

def _error_LT(y: list, z: list, tow_number)->list:
    """"This function takes a given tow path
        and calculates the error between the
        actual path and the intended path"""
    
    error_y = []
    error_z = []

    if tow_number == 1:
        y_ref = 125
    else:
        y_ref = 125 + 12.5 * (tow_number-2)

    for i in range(len(y)):
        error_y.append(y[i] - y_ref)
        error_z.append(z[i] - z_ref)

    return error_y, error_z

################################################################################################################
"""Functions for Laser Line Scanner"""

def _handle_LLS(time: list, left_edge: list, right_edge: list, width:list) -> pd.DataFrame:
    """"This function takes the processed data and
        creates new data points for each time stamp
        where each point in time has a corresponding
        width and its the center of the tow"""
    

    
    rows = len(time)
    columns = 4
    shape = (rows, columns)
    pandas_table = np.empty(shape)
    zero_time = time_to_float(time[0])

    for i in range(len(time)):
        pandas_table[i][0] = time_to_float(time[i]) - zero_time
        pandas_table[i][1] = width[i] # width
        pandas_table[i][2] = 0.5*(right_edge[i] + left_edge[i]) # center
        pandas_table[i][3] = (pandas_table[i][1]-6.35) # error (6.35 is the right width)
    
    pandas_table = pd.DataFrame(pandas_table)
    pandas_table.columns = ["time", "width", "center","width error"]

    return pandas_table

################################################################################################################
"""Functions for Camera"""

def _handle_camera(time: list, left_edge: list, right_edge: list, width:list) -> pd.DataFrame:

    rows = len(time)
    columns = 4
    shape = (rows, columns)
    pandas_table = np.empty(shape)
    zero_time = time_to_float(time[0])

    # Sign flipped since camera is flipped
    for i in range(len(time)):
        pandas_table[i][0] = time_to_float(time[i]) - zero_time
        pandas_table[i][1] = width[i] # width
        pandas_table[i][2] = 0.5*(right_edge[i] + left_edge[i]) # center
        pandas_table[i][3] = abs(pandas_table[i][1] - (-6.35))
    
    pandas_table = pd.DataFrame(pandas_table)
    pandas_table.columns = ["time", "width", "center","width error"]

    return pandas_table

################################################################################################################
'''Extra functions that might be needed'''

def time_to_float(date:str)->float:
    """converts a string into a float of time"""
    date = date.strip("'").split(" ")[1]
    hour, minute, second = date.split(":")
    return float(second) + float(minute) * 60 + float(hour) * 3600

def convert_coordinates(start:tuple,end:tuple, coord:tuple)->tuple:
    '''This function converts the coordinate into a new
        coordinate system based on the line between start and end'''

    vector = np.array(end) - np.array(start) # a vector between start and end

    unit = vector / vector.dot(vector) # the unit vector in that direction

    normal = np.rot90(unit)

    proj_tangent = unit.dot(coord) # gets the projection. I.e. the coordinates in the new system
    proj_normal = normal.dot(coord)

    return proj_tangent, proj_normal 

################################################################################################################
"""Functions for purging, saving and loading data"""

_save_path = "Processed data\\"

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

def get_processed_data(tow:int, sensor_type:str, overwrite=False, helper=False)->pd.DataFrame:
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

def get_synced_data(tow:int, spacesynced:bool = False, overwrite:bool=False)->pd.DataFrame:
    "work in progress"

def main():
    "Hi"

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else
