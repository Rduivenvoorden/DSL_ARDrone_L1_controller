import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
import csv
from os import listdir
from os import path


show_plots = False
interactive = False
save_imgs = True

compare_L1_type = 2
compare_LPF_type = 1
#compare_Gamma_value = 1000


### Evaluate the cost function given the log file filename
def determineXYZCost(filename):
  
  print filename

  ### Open filename ###
  data = np.genfromtxt(filename, delimiter = ',')

  ### Extract data values ###
  t = data[:,0] + data[:,1]*(10.0**(-9.0))
  t = t-t[0]

  #
  x_ref = data[:,2:5] ### reference output predictor output
  x_dot = data[:,5:8] ### output

  sigma_hat = data[:,8:11] ### parameter estimate

  x_L1_des = data[:,11:14] ### system input

  L1_input = data[:,14:17] ### L1 input
  x = data[:,17:20] ### output

  x_des = data[:,20:23] ### input

  desired_acc = data[:,23:26] ### only for L1 type 3
  rpy = data[:,26:29] ### output

  if L1_type == 4:
    nav_rp = data[:,29:31]

  ### Determine at which indices the desired reference trajectory changes (assuming step inputs)
  ref_signal_changes = np.zeros(50)
  old_xdes = x_des[0][0]
  des_counter = 0

  for index in range(0,t.shape[0]):

    if not old_xdes == x_des[index][0]:
      ref_signal_changes[des_counter] = index
      des_counter = des_counter + 1
      old_xdes = x_des[index][0]

    # end if
  # end for
  ref_signal_changes = ref_signal_changes[0:des_counter]

  if max(ref_signal_changes.shape) <=2:
    return -1

  ### Save the desired position and the actual position for the second and subsequent changes until the last change (for all directions)
  err_calc_xdes = x_des[ int(ref_signal_changes[1]):int(ref_signal_changes[-1]) , :]
  err_calc_x = x[ int(ref_signal_changes[1]):int(ref_signal_changes[-1]) , :]
  err_calc_t = t[ int(ref_signal_changes[1]):int(ref_signal_changes[-1])]

  x_track_error = err_calc_xdes[:,0] - err_calc_x[:,0]
  y_track_error = err_calc_xdes[:,1] - err_calc_x[:,1]
  z_track_error = err_calc_xdes[:,2] - err_calc_x[:,2]

  x_error_metric = np.mean(x_track_error**2)
  y_error_metric = np.mean(y_track_error**2)
  z_error_metric = np.mean(z_track_error**2)

  ux_metric = np.mean(x_L1_des[:,0]**2)
  uy_metric = np.mean(x_L1_des[:,1]**2)
  uz_metric = np.mean(x_L1_des[:,2]**2)

  cost = np.sqrt(x_error_metric + y_error_metric + z_error_metric) + np.sqrt(ux_metric + uy_metric + uz_metric)

  #print "\nCost: ", cost, "\n"
  return cost

# end def

###############################################################################

# open the file from input argument from user
print "\nOpening the following: ", sys.argv[1]#[26:45]

file_input = sys.argv[1]
#data = np.genfromtxt(file_input, delimiter = ',')

info_input = file_input[:45] + '_l1_experiment_info.csv'

directory = sys.argv[1][:26]


CostFunction = np.zeros((500,3))


### Loop through all files in the directory
logfile_counter = 0

for log_file in listdir(directory):

  experiment = log_file[0:19]

  full_log_file = directory + log_file

  # check whether file is actually experiment info file
  if not full_log_file.find("_info") == -1:

    with open(full_log_file, 'rb') as f:
      reader = csv.reader(f)
      experiment_info_header = reader.next()
      #print experiment_info
      experiment_info_values = reader.next()
      #print experiment_info



    L1_type = int(float(experiment_info_values[0]))
    print L1_type

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

    #LPF_type = int(float(experiment_info_values[14]))

    # Check if experiment data exists for this experiment
    if L1_type == compare_L1_type and path.isfile(directory + experiment + "_l1_ref_output.csv"):

      # Then evaluate cost and increase file counter
      cost = determineXYZCost(directory + experiment + "_l1_ref_output.csv")
      
      if not cost == -1:
        CostFunction[logfile_counter,0] = om_xy      
        CostFunction[logfile_counter,1] = m_xy     
        CostFunction[logfile_counter,2] = cost

        logfile_counter = logfile_counter + 1

    # end if
  # end if
  
# end for

CostFunction = CostFunction[0:logfile_counter,:]

###############################################################################

print CostFunction

omega = np.unique(CostFunction[:,0])
refmodel = np.unique(CostFunction[:,1])

print omega
print refmodel

X, Y = np.meshgrid(omega, refmodel) 

Z = np.zeros( (max(omega.shape), max(refmodel.shape)) )

for cost_index in range(0,max(CostFunction.shape)):

  om = CostFunction[cost_index][0]
  ref = CostFunction[cost_index][1]

  Z_x_index = np.where(omega == om)[0][0]
  Z_y_index = np.where(refmodel == ref)[0][0]

  Z[Z_x_index][Z_y_index] = CostFunction[cost_index][2]

# end for

print Z

print "\n"

for i in range(0,Z.shape[0]):
  for j in range(0,Z.shape[1]):
    print Z[i][j],

  print ""
  
