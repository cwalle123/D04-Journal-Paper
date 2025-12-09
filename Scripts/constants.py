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

# programmed y_offset for lay-up
y_offset_layup = 125.5 #mm

# programmed y-offset between consecutive tows
y_increment_programmed = 12.5 #mm

# programmed y-offset for traverse gap data collection
y_offset_traverse = 125 #mm

# programmed increment size for traverse gap data collection
y_increment_traverse = 12.5 #mm

# used frame width of LLS B for traverse gap data collection
frame_width_traverse = 12.5 #mm

##############################################################################################################
"""Figure Formatting"""

font_TNR = "Times New Roman"
font_label = 10         #points
font_axis_ticks = 8     #points
font_legend = 10        #points

figure_width = 345 / 72.27                      #points/points per inch = inch, amount of points taken from \the\linewidth in Overleaf
min_figure_height = 0.313 * figure_width        #Aspect ratio (heigt:width) of shortest figure (figure 8) * figure width. Figure heights should be multiples of this value, then Latex will take care of the scaling
tick_length = 5                                 #points
tick_width = 1.2                                #points
graph_box_thickness = 1                         #points

color_exp = "blue"  # AKA Experiment
color_RW = "green"  # AKA MCMC
color_RS = "orange" # AKA MC

##############################################################################################################
"""Model Parameters"""

# Consecutive_Error_Bins = 470
# number_of_steps = 2440
Consecutive_Error_Bins = 150
number_of_steps = 370
NOMINAL_LT_Y = 0  # nominal value for LT y
NOMINAL_LLS_A = 6.35  # nominal value for LLS A
NOMINAL_LLS_B = 6.35  # nominal value for LLS B
NOMINAL_CAM = 0  # nominal value for CAM

# steps=868, bins=80
# steps=825, bins=90