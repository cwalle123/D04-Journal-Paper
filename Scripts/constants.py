'''This file is to store constants that are useful throughout the entire project'''

##############################################################################################################
"""Expirmental set up"""

#Dimensions of the set-up
roller_width = 31 #mm
roller_diameter = 40 #mm

# Reference distances between sensors, positive value is ahead of center point on the tow
TCP_LLS_A = -310.45 #mm
TCP_LLS_B = 107 #mm
TCP_CAM = -roller_diameter * 3.1415926 / 4

##############################################################################################################
"""Tow geometry"""

# reference coordinates for calculating error
z_ref = -4  # mm

# specified tow witdh
tow_width_specified = 6.35 #mm

# programmed y-offset between consecutive tows
y_increment_programmed = 12.5 #mm

##############################################################################################################
"""Figure Formatting"""

font_extra_small = 8
font_small = 12
font_medium = 14
font_large = 16
font_extra_large = 24

##############################################################################################################
"""Model Parameters"""

Consecutive_Error_Bins = 40
NOMINAL_LT_Y = 0  # nominal value for LT y
NOMINAL_LLS_A = 6.25  # example nominal value for LLS A
NOMINAL_LLS_B = 6.35  # nominal value for LLS B
NOMINAL_CAM = 0  # nominal value for CAM