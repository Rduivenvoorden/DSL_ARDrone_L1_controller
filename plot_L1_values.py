import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import csv


interactive = 1

# open the file from input argument from user
print "\nOpening the following: ", sys.argv[1]#[26:45]

file_input = sys.argv[1]
data = np.genfromtxt(file_input, delimiter = ',')

info_input = file_input[:45] + '_l1_experiment_info.csv'

###############################################################################

with open(info_input, 'rb') as f:
  reader = csv.reader(f)
  experiment_info = reader.next()
  experiment_info = reader.next()

print '\nL1_type: ', experiment_info[0]
L1_type = int(float(experiment_info[0]))


if L1_type == 1:
  print "\nL1 x-y-z position controller with piecewice constant adaptation"

  print 'Cutoff Frequency (x,y,z): ', experiment_info[1][:5], experiment_info[2], experiment_info[3]
  print 'Output Predictor (x,y,z): ', experiment_info[4], experiment_info[5], experiment_info[6]
  print 'Adaptive Gain (Gamma): ', experiment_info[13]

elif L1_type == 2:
  print "\nL1 x-y-z translational velocity controller with projection based adaptation"

  print 'Cutoff Frequency (x,y,z): ', experiment_info[1][:5], experiment_info[2][:5], experiment_info[3][:5]
  print 'Output Predictor (x,y,z): ', experiment_info[4][:5], experiment_info[5][:5], experiment_info[6][:5]
  print 'P gain: ', experiment_info[7][:5], experiment_info[8][:5], experiment_info[9][:5]
  print 'Adaptive Gain (Gamma): ', experiment_info[13][:5]

elif L1_type == 3:
  print "L1 roll-pitch angle, z velocity controller with projection based adaptation"
  
  print 'Cutoff Frequency (x,y,z): ', experiment_info[1][:5], experiment_info[2][:5], experiment_info[3][:5]
  print 'Output Predictor (x,y,z): ', experiment_info[4][:5], experiment_info[5][:5], experiment_info[6][:5]
  print 'P gain: ', experiment_info[7][:5], experiment_info[8][:5], experiment_info[9][:5]
  print 'D gain: ', experiment_info[10][:5], experiment_info[11][:5], experiment_info[12][:5]
  print 'Adaptive Gain (Gamma): ', experiment_info[13][:5]

elif L1_type == 4:
  print "L1 augmented standard nonlinear controller with projection based adaptation"

  print 'Cutoff Frequency (x,y,z): ', experiment_info[1][:5], experiment_info[2][:5], experiment_info[3][:5]
  print 'Output Predictor (x,y,z): ', experiment_info[4][:5], experiment_info[5][:5], experiment_info[6][:5]
  print 'Adaptive Gain (Gamma): ', experiment_info[13][:5]

else:
  print "Standard DSL PD nonlinear controller"
  # No L1 parameters to define#

###############################################################################

##### Extract Data #####

#writer.writerow(np.array([now.secs, now.nsecs, self.x_ref[0][0], self.x_ref[1][0], self.x_ref[2][0], curr.x_dot[0], curr.x_dot[1], curr.x_dot[2], self.sigma_hat[0][0], self.sigma_hat[1][0], self.sigma_hat[2][0], self.x_L1_des[0][0], self.x_L1_des[1][0], self.x_L1_des[2][0], self.desired_vel[0][0], self.desired_vel[1][0], self.desired_vel[2][0], curr.x[0], curr.x[1], curr.x[2], des.x[0], des.x[1], des.x[2], desired_acc[0][0], desired_acc[1][0], desired_acc[2][0], curr.rpy[0], curr.rpy[1], curr.rpy[2]]))

exp_time = data[:,0] + data[:,1]*(10**(-9))
exp_time = exp_time-exp_time[0]

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

#ax = data[:,29]
#ay = data[:,30]
#
#ax_b = data[:,31]
#ay_b = data[:,32]

###############################################################################

if not(interactive):
  plt.ion()

plt.figure(1)
plt.plot(exp_time,x_des[:,0])
plt.plot(exp_time,x[:,0])
plt.xlabel('Time (sec)')
plt.ylabel('Position (m)')
plt.legend(['Desired x', 'Actual x'])
#plt.title('')
plt.axis([plt.axis()[0], plt.axis()[1], -2.0, 2.0])
plt.grid(True)
#plt.show()


plt.figure(2)
plt.plot(exp_time,x_des[:,1])
plt.plot(exp_time,x[:,1])
plt.xlabel('Time (sec)')
plt.ylabel('Position (m)')
plt.legend(['Desired y', 'Actual y'])
#plt.title('')
plt.axis([plt.axis()[0], plt.axis()[1], -2.0, 2.0])
plt.grid(True)
#plt.show()


plt.figure(3)
plt.plot(exp_time,x_des[:,2])
plt.plot(exp_time,x[:,2])
plt.xlabel('Time (sec)')
plt.ylabel('Position (m)')
plt.legend(['Desired z', 'Actual z'])
#plt.title('')
plt.axis([plt.axis()[0], plt.axis()[1], 0, 2.5])
plt.grid(True)
#plt.show()

###############################################################################

plt.figure(4)
plt.plot(exp_time,sigma_hat[:,0], exp_time,sigma_hat[:,1], exp_time,sigma_hat[:,2])
plt.xlabel('Time (sec)')
plt.ylabel('$\hat{\sigma}$')
plt.legend(['$\sigma_x$', '$\sigma_y$', '$\sigma_z$'])
#plt.title('')
#plt.axis([plt.axis()[0], plt.axis()[1], 0, 1.5])
plt.grid(True)
#plt.show()

###############################################################################

###

t = exp_time
#MED_input = L1_input[:,0]*180/np.pi - sigma_hat[:,0]*180/np.pi
#LPF_input = L1_input[:,0]*180/np.pi - sigma_hat[:,0]*180/np.pi
LPF_input = L1_input[:,0] - sigma_hat[:,0]
#LPF_input = np.zeros(t.shape[0])
LPF_output = np.zeros(t.shape[0])


### First Third order Median Filter the input signal

in_old = 0
in_oldold = 0

#for index in range(0,t.shape[0]):
#  
#  LPF_input[index] = np.median(np.array([in_oldold, in_old, MED_input[index]]))
#  in_oldold = in_old
#  in_old = MED_input[index]



u_old = 0
y_ddot = 0
y_dot = 0
y = 0

omega_cutoff = 1.75

old_time = 0

for index in range(0,t.shape[0]):

  dt = t[index] - old_time
  if not(dt == 0):
    old_time = t[index]
  
    # first find derivative of input signal (i.e. u = track_error, u_dot = d/dt(track_error) )
    u_dot = 1/dt*(LPF_input[index] - u_old) # u_dot = 1/dt*(u - u_old)
    u_old = LPF_input[index] # set current u to track_error (in next iteration, this is automatically u_old)
    
    y_ddot = y_ddot + dt*(-3*omega_cutoff*(y_ddot) - 3*(omega_cutoff**2)*(y_dot) - (omega_cutoff**3)*(y) + 3*(omega_cutoff**2)*(u_dot) + (omega_cutoff**3)*(LPF_input[index]) )
    y_dot = y_dot + dt*(y_ddot)
    y = y + dt*(y_dot)
  
    LPF_output[index] = y
  else:
    print "dt is zero"
    LPF_output[index] = 0


plt.figure(10)
#plt.plot(t,MED_input)
plt.plot(t,LPF_input)
plt.plot(t,LPF_output)
plt.plot(exp_time, x_L1_des[:,0])
#plt.plot(t,ax)
#plt.plot(t,ay)
#plt.plot(t,ax_b)
#plt.plot(t,ay_b)
plt.grid(True)

###





plt.figure(5)
if L1_type == 1:
  print L1_type

elif L1_type == 2:
  plt.plot(exp_time, x_dot[:,0])
  plt.plot(exp_time, x_L1_des[:,0])
  plt.plot(exp_time, L1_input[:,0])
  plt.plot(exp_time, L1_input[:,0] - sigma_hat[:,0])
  plt.plot(exp_time, x_ref[:,0])
  plt.xlabel('Time (sec)')
  plt.ylabel('Velocity (m)')
  plt.legend(['x-dot','x-L1','x-dot-des','x-dot-des -- sigma-hat','x-dot-ref'])

elif L1_type == 3:
  plt.legend(['x-dot','x-dot-L1','x-dot-des','x-dot-des -- sigma-hat'])

elif L1_type == 4:
  plt.plot(exp_time, rpy[:,0]*180/np.pi)
  plt.plot(exp_time, x_L1_des[:,0]*180/np.pi)
  plt.plot(exp_time, L1_input[:,0]*180/np.pi)
  plt.plot(exp_time, L1_input[:,0]*180/np.pi - sigma_hat[:,0]*180/np.pi)
  plt.plot(exp_time, x_ref[:,0]*180/np.pi)
  #plt.plot(t,LPF_output)
  plt.xlabel('Time (sec)')
  plt.ylabel('Roll Angle (deg)')
  plt.legend(['Roll','Roll-L1-des','Roll-des','Roll-des -- sigma-hat','Roll-ref'])
  #plt.axis([plt.axis()[0], plt.axis()[1], -30.0, 30.0])

#plt.title('')
#plt.axis([plt.axis()[0], plt.axis()[1], -2.0, 2.0])
plt.grid(True)
#plt.show()


plt.figure(6)
if L1_type == 1:
  print L1_type

elif L1_type == 2:
  plt.plot(exp_time, x_dot[:,1])
  plt.plot(exp_time, x_L1_des[:,1])
  plt.plot(exp_time, L1_input[:,1])
  plt.plot(exp_time, L1_input[:,1] - sigma_hat[:,1])
  plt.plot(exp_time,x_ref[:,1])
  plt.xlabel('Time (sec)')
  plt.ylabel('Velocity (m)')
  plt.legend(['y-dot','y-L1','y-dot-des','y-dot-des -- sigma-hat','y-dot-ref'])

elif L1_type == 3:
  plt.legend(['y-dot','y-dot-L1','y-dot-des','y-dot-des -- sigma-hat'])

elif L1_type == 4:
  plt.plot(exp_time, rpy[:,1]*180/np.pi)
  #plt.plot(exp_time, x_ref[:,1]*180/np.pi)
  plt.plot(exp_time, x_L1_des[:,1]*180/np.pi)
  plt.plot(exp_time, L1_input[:,1]*180/np.pi)
  plt.plot(exp_time, L1_input[:,1]*180/np.pi - sigma_hat[:,1]*180/np.pi)
  plt.plot(exp_time, x_ref[:,1]*180/np.pi)
  plt.xlabel('Time (sec)')
  plt.ylabel('Pitch Angle (deg)')
  plt.legend(['Pitch','Pitch-L1-des','Pitch-des','Pitch-des -- sigma-hat','Pitch-ref'])
  #plt.legend(['Pitch','Pitch_ref','Pitch-L1-des','Pitch-des','Pitch-des -- sigma-hat'])
  #plt.axis([plt.axis()[0], plt.axis()[1], -30.0, 30.0])

#plt.title('')
#plt.axis([plt.axis()[0], plt.axis()[1], -2.0, 2.0])
plt.grid(True)
#plt.show()


plt.figure(7)
plt.plot(exp_time, x_dot[:,2])
plt.plot(exp_time, x_L1_des[:,2])
plt.plot(exp_time, L1_input[:,2])
plt.plot(exp_time, L1_input[:,2] - sigma_hat[:,2])
plt.plot(exp_time,x_ref[:,2])
plt.xlabel('Time (sec)')
plt.ylabel('Velocity (m/s)')
plt.legend(['z-dot','z-dot-L1','z-dot-des','z-dot-des -- sigma-hat'])
#plt.title('')
#plt.axis([plt.axis()[0], plt.axis()[1], -1.0, 1.0])
plt.grid(True)

if not(interactive):
  plt.figure(8)
  plt.close(8)
  time.sleep(30)

else:
  plt.show()

###############################################################################

#plt.close('all')
print('')
