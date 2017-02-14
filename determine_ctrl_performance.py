#!/usr/bin/env python2

"""Script for plotting the cost function of the L1 adaptive controller.

Loops through all CSV log files of experiments where low pass filter cutoff
frequency and output predictor eigenvalue are changed between iterations of an
L1 adaptive controller. This script produces Figure 7.10 in:

http://hdl.handle.net/1807/74674

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
import csv
from os import listdir
from os import path

# Set flags for plots and figure saving
show_plots = False
interactive = False
save_imgs = True

compare_L1_type = 2

# Evaluate the cost function given the log file filename
def determineXYZCost(filename):

    # Open filename
    data = np.genfromtxt(filename, delimiter = ',')

    # Extract data values
    t = data[:, 0] + data[:, 1] * (10.0 ** -9)
    t = t - t[0]

    x_ref = data[:, 2:5] # reference output predictor output
    x_dot = data[:, 5:8] # output

    sigma_hat = data[:, 8:11] # parameter estimate

    x_L1_des = data[:, 11:14] # system input

    L1_input = data[:, 14:17] # L1 input
    x = data[:, 17:20] # output

    x_des = data[:, 20:23] # input

    desired_acc = data[:, 23:26] # only for L1 type 3
    rpy = data[:, 26:29] # output

    if L1_type == 4:
        nav_rp = data[:, 29:31]

    
    ref_signal_changes = np.zeros(50)
    old_xdes = x_des[0][0]
    des_counter = 0

    # Determine at which indices the desired reference trajectory changes
    # (assuming step inputs)
    for index in range(0,t.shape[0]):

        if not old_xdes == x_des[index][0]:
            ref_signal_changes[des_counter] = index
            des_counter += 1
            old_xdes = x_des[index][0]

    # Truncate the vector containing indices of changes in reference signal
    ref_signal_changes = ref_signal_changes[0:des_counter]

    # Return -1 if there is insufficient data in the log file to compute the
    # cost function accurately
    if max(ref_signal_changes.shape) <=2:
        return -1

    print filename

    # Save the desired position and the actual position for the second and 
    # subsequent changes until the last change (for all directions).
    err_calc_xdes = x_des[int(ref_signal_changes[1]):int(
                          ref_signal_changes[-1]), :]
    err_calc_x = x[int(ref_signal_changes[1]):int(ref_signal_changes[-1]), :]
    err_calc_t = t[int(ref_signal_changes[1]):int(ref_signal_changes[-1])]

    x_track_error = err_calc_xdes[:, 0] - err_calc_x[:, 0]
    y_track_error = err_calc_xdes[:, 1] - err_calc_x[:, 1]
    z_track_error = err_calc_xdes[:, 2] - err_calc_x[:, 2]

    x_error_metric = np.mean(x_track_error ** 2)
    y_error_metric = np.mean(y_track_error ** 2)
    z_error_metric = np.mean(z_track_error ** 2)

    ux_metric = np.mean(x_L1_des[:, 0] ** 2)
    uy_metric = np.mean(x_L1_des[:, 1] ** 2)
    uz_metric = np.mean(x_L1_des[:, 2] ** 2)

    # Compute the cost
    cost = np.mean(3.0 * np.sqrt(x_track_error ** 2 + y_track_error ** 2 + 
                   z_track_error ** 2) + np.sqrt(ux_metric ** 2 + 
                   uy_metric ** 2 + uz_metric ** 2))

    return cost


# open the file from input argument from user
print "\nOpening the following: ", sys.argv[1]#[26:45]

directory = sys.argv[1][:26]

# Initialize the cost function value matrix
CostFunction = np.zeros((500,3))

# Loop through all files in the directory
logfile_counter = 0

for log_file in listdir(directory):

    experiment = log_file[0:19]

    full_log_file = directory + log_file

    # check whether file is actually experiment info file
    if not full_log_file.find("_info") == -1:

        # Open the current log file for reading
        with open(full_log_file, 'rb') as f:
            reader = csv.reader(f)
            experiment_info_header = reader.next()
            print experiment
            experiment_info_values = reader.next()

        # Extract the experiment iteration parameters
        L1_type = int(float(experiment_info_values[0]))

        om_xy = float(experiment_info_values[1])
        om_y = float(experiment_info_values[2])
        om_z = float(experiment_info_values[3])

        m_xy = float(experiment_info_values[4])
        m_y = float(experiment_info_values[5])
        m_z = float(experiment_info_values[6])

        P_xy = float(experiment_info_values[7])
        P_y = float(experiment_info_values[8])
        P_z = float(experiment_info_values[9])

        Gamma = int(float(experiment_info_values[13]))

        # Check if the experiment is valid for comparison, and if log data 
        # exists for this experiment
        if L1_type == compare_L1_type and path.isfile(directory + experiment + 
                                                  "_l1_ref_output.csv"):

            # Then evaluate cost and increase file counter
            cost = determineXYZCost(directory + experiment +
                                    "_l1_ref_output.csv")

            # If the log file contained meaningful data, store the cost
            if not cost == -1:
                CostFunction[logfile_counter,0] = om_xy      
                CostFunction[logfile_counter,1] = m_xy     
                CostFunction[logfile_counter,2] = cost

                logfile_counter = logfile_counter + 1

# Truncate the cost function value matrix 
CostFunction = CostFunction[0:logfile_counter,:]

print CostFunction

# Plot and save the cost function for each parameters pair 
omega = np.unique(CostFunction[:,0])
refmodel = np.unique(CostFunction[:,1])

# Create a meshgrid
X, Y = np.meshgrid(omega, refmodel) 

Z = np.zeros((max(refmodel.shape), max(omega.shape)))
multiples = np.zeros((max(refmodel.shape), max(omega.shape)))

# Populate the Z matrix with cost function data for plotting
for cost_index in range(0,max(CostFunction.shape)):

    om = CostFunction[cost_index][0]
    ref = CostFunction[cost_index][1]

    Z_y_index = np.where(refmodel == ref)[0][0]
    Z_x_index = np.where(omega == om)[0][0]

    # Add the cost for the parameter pair into the correct index (adding for 
    # averaging later)
    Z[Z_y_index][Z_x_index] = (Z[Z_y_index][Z_x_index] + 
                               CostFunction[cost_index][2])

    # Keep track of the total number of entries for each parameter pair
    multiples[Z_y_index][Z_x_index] = multiples[Z_y_index][Z_x_index] + 1

# Fill empty indices in Z with 'nan'
empty_indices = np.where(multiples == 0)

for i in range(0, empty_indices[0].shape[0]):
    multiples[empty_indices[0][i]][empty_indices[1][i]] = float('nan')

# Take the average of the cost function for multiple data
Z = Z/multiples

# Print the cost function values for possible plotting in MATLAB
print "\nZ = [",

for i in range(0,Z.shape[0]):
    for j in range(0,Z.shape[1]):
        if not j == Z.shape[1]-1:
            print Z[i][j], ",",
        else:
            print Z[i][j],

    if not i == Z.shape[0]-1:
        print ";"
    else:
        print "];"

levels = np.arange(np.min(Z), np.max(Z), 0.01)
norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
cmap = cm.PRGn

plt.figure(1, figsize=(9,6))
CS = plt.contourf(X, Y, Z, levels, cmap=plt.cm.jet)
cbar = plt.colorbar(CS, format="%.3f")
cbar.ax.set_ylabel('Cost Function')

plt.xlabel('Cut-off Frequency [rad/s]')
plt.ylabel('Output Predictor Eigenvalue')
plt.show()
