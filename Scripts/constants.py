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

figure_width = 345 / 72.27                      #points/points per inch = inch, amount of points taken from \the\textwidth in Overleaf
min_figure_height = 0.382 * figure_width        #Aspect ratio (heigt:width) of shortest figure (figure 8) * figure width. Figure heights should be multiples of this value, then Latex will take care of the scaling
graph_line_thickness = 1                        #points
legend_line_thickness = 2                       #points
annotation_thickness = 0.75                     #points
annotation_stripe_height = 0.04                 #points
tick_width = 0.5                                #points
tick_length = 3                                 #points
graph_box_thickness = 0.5                       #points
legend_box_thickness = graph_box_thickness      #points, has to be the same as graph box thickness

color_exp = (0, 0.4470, 0.7410)                 # AKA Experiment, blue
color_RW = (0.22, 0.51, 0.35)                   # AKA MCMC, green
color_RS = (0.8, 0.7, 0.2)                      # AKA MC, orange
color_annotations = (0.8500, 0.3250, 0.0980)    # axis annotations figure 2, redish
color_PDF_fits = (0.9290, 0.6940, 0.1250)       # PDF color figure 2, orange
color_borders = (0, 0, 0)                       # graph boxes, legends etc.
color_ideal_gap = (0.5, 0.5, 0.5)               # for figure 7, gray
transparency = 0.8                              # transparency of shaded areas

legend_space = 0.18

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

def main():
    print(figure_width)

if __name__ == "__main__":
    main() # makes sure this only runs if you run *this* file, not if this file is imported somewhere else