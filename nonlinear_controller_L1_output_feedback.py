#!/usr/bin/env python

# PURPOSE
# This controller takes in current state information through the /current_coordinates topic and requests waypoint information from the /path_follower node for the desired state. Based on the difference between these two states, the controller computes outputs and sends commands to the drone.
# This controller also has a keypress handler to allow for manual control of the vehicle.

# SUBSCRIBED TOPICS
# /current_coordinates
# /path_coordinates
# /ardrone/navdata

# PUBLISHED TOPICS
# /cmd_vel_ideal [rad]
# /ardrone/land 
# /ardrone/takeoff
# /ardrone/reset
# /waypoint_request


####################
# IMPORT LIBRARIES #
####################

# Import ROS libraries, rospy, and load manifest file for access to project dependencies
import rospy
import roslib; roslib.load_manifest('dsl__controllers__basic')

import time
import sys
import math
import numpy as np

import csv

# Load the DroneVideoDisplay class, which handles video display
from drone_video_display import DroneVideoDisplay
from drone_status import DroneStatus

# The GUI libraries
from PySide import QtCore, QtGui


###################
# IMPORT MESSAGES #
###################

from dsl__utilities__msg.msg import StateVector
from dsl__utilities__msg.msg import StateData
from ardrone_autonomy.msg import Navdata
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty  
from std_msgs.msg import Bool
from std_msgs.msg import Float32

from rospy.numpy_msg import numpy_msg


# For shutting down the QT application window in CTRL-C
import signal
from PyQt4.QtCore import QTimer
from PyQt4.QtGui import QApplication, QMessageBox
def sigint_handler(*args):
    """Handler for the SIGINT signal."""
    QApplication.quit()


##################       Key Mapping Object     ################################
class KeyMapping(object):
  ForwardLeft      = QtCore.Qt.Key.Key_W
  Forward          = QtCore.Qt.Key.Key_E
  ForwardRight     = QtCore.Qt.Key.Key_R
  Right            = QtCore.Qt.Key.Key_F
  BackwardRight    = QtCore.Qt.Key.Key_V
  Backward         = QtCore.Qt.Key.Key_C
  BackwardLeft     = QtCore.Qt.Key.Key_X
  Left             = QtCore.Qt.Key.Key_S
  YawLeft          = QtCore.Qt.Key.Key_A
  YawRight         = QtCore.Qt.Key.Key_G
  IncreaseAltitude = QtCore.Qt.Key.Key_Q
  DecreaseAltitude = QtCore.Qt.Key.Key_Z
  Takeoff          = QtCore.Qt.Key.Key_Y
  Land             = QtCore.Qt.Key.Key_H
  Emergency        = QtCore.Qt.Key.Key_Space
  StartHover       = QtCore.Qt.Key.Key_I
  EndHover         = QtCore.Qt.Key.Key_K
  GoInitPos        = QtCore.Qt.Key.Key_O # Custom functions
  StartExp         = QtCore.Qt.Key.Key_P # 
  LandAll          = QtCore.Qt.Key.Key_J
  TakeoffAll       = QtCore.Qt.Key.Key_U
  SaveToFile	   = QtCore.Qt.Key.Key_T #My functions to dump into csv file
  ReadFromFile     = QtCore.Qt.Key.Key_B #Also mine


#####################   Useful Structures for the Controller ###################

class State:
  x      = np.array([0,0,0])
  x_dot  = np.array([0,0,0])
  x_ddot = np.array([0,0,0])

  rpy    = np.array([0,0,0],dtype=np.float64)

  def __init__(self, x_     =np.array([0,0,0]),
                     x_dot_ =np.array([0,0,0]),
                     x_ddot_=np.array([0,0,0]),
                     rpy_   =np.array([0,0,0])):
    self.x      = x_
    self.x_dot  = x_dot_
    self.x_ddot = x_ddot_
    self.rpy    = rpy_

  @classmethod
  def fromState(cls, ss):
    new_class = cls(ss.x, ss.x_dot, ss.x_ddot, ss.rpy)
    return new_class

class DroneCommand:
  roll    = 0
  pitch   = 0
  yaw_dot = 0
  z_dot   = 0

  twist = Twist()

class Status:
  drone_state       = -1 # hover/flight etc
  keyboard_override = 0
  hover             = 0
  request_waypoint  = Bool(True)
  t_last_cmd        = 0


#####################   Main Controller Code    ################################

class DroneController(DroneVideoDisplay):

  # Member Variables
  current_state = State()
  desired_state = State()
  command       = DroneCommand()
  status        = Status()
  
  # some parameters
  tau_x = 0.6 #original was 1.5
  tau_y = 0.6 #original was 1.5
  tau_z = 0.8
  tau_w = 0.7
  zeta  = 0.707 #original was 0.707
  
  g = 9.81
  max_euler = 0.
  max_vz    = 0.
  max_yaw   = 0.

  #*****************************************************************************

  # Constructor
  def __init__(self):
    super(DroneController,self).__init__()
    rospy.loginfo('Initializing Non-linear Controller---------------------------')

    # Subscribers 
    # path_coordinates = desired coordinates
    # current_coordinates = vicon coordinates
    # Publishers
    self.pubLand    = rospy.Publisher('ardrone/land', Empty)
    self.pubTakeoff = rospy.Publisher('ardrone/takeoff', Empty)
    self.pubReset   = rospy.Publisher('ardrone/reset', Empty)
    self.pubCommand = rospy.Publisher('cmd_vel_ideal', Twist)
    self.pubGoInitPos = rospy.Publisher('/go_init_pos', Empty)
    self.pubStartExp = rospy.Publisher('/start_exp', Empty)
    self.pubLandAll = rospy.Publisher('/land_all', Empty)
    self.pubTakeoffAll = rospy.Publisher('/takeoff_all', Empty)
    self.pubSaveToFile = rospy.Publisher('/save_to_file', Empty)
    self.pubReadFromFile = rospy.Publisher('/read_from_file',Empty)
    
    # Publish L1 data
    self.pubParama = rospy.Publisher('ardrone/parama',Float32)
    self.pubParamb = rospy.Publisher('ardrone/paramb',Float32)
    self.pubParamc = rospy.Publisher('ardrone/paramc',Float32)
    self.pubL1des_x = rospy.Publisher('ardrone/l1des_x',Float32)
    self.pubL1des_y = rospy.Publisher('ardrone/l1des_y',Float32)
    self.pubL1des_z = rospy.Publisher('ardrone/l1des_z',Float32)
    self.pubXref_x = rospy.Publisher('ardrone/xref_x',Float32)
    self.pubXref_y = rospy.Publisher('ardrone/xref_y',Float32)
    self.pubXref_z = rospy.Publisher('ardrone/xref_z',Float32)
    self.pubXcurr_x = rospy.Publisher('ardrone/xcurr_x',Float32)
    self.pubXcurr_y = rospy.Publisher('ardrone/xcurr_y',Float32)
    self.pubXcurr_z = rospy.Publisher('ardrone/xcurr_z',Float32)
    self.pubXdotcurr_x = rospy.Publisher('ardrone/xdotcurr_x',Float32)
    self.pubXdotcurr_y = rospy.Publisher('ardrone/xdotcurr_y',Float32)
    self.pubXdotcurr_z = rospy.Publisher('ardrone/xdotcurr_z',Float32)

    self.sub_cur     = rospy.Subscriber('estimated_state', StateVector, self.updateCurrentState)
    self.sub_navdata = rospy.Subscriber('ardrone/navdata', Navdata, self.updateNavdata)
    self.sub_des     = rospy.Subscriber('path_coordinates', StateData, self.updateDesiredState)
    self.sub_land_all = rospy.Subscriber('/land_all', Empty, self.SendLand)    
    self.sub_takeoff_all = rospy.Subscriber('/takeoff_all', Empty, self.SendTakeoff)

    # Control Parameters
    self.max_euler = rospy.get_param('ardrone_driver/euler_angle_max', 0.26)  # rads
    self.max_vz    = rospy.get_param('ardrone_driver/control_vz_max', 1.0)  # m/sec
    self.max_yaw   = rospy.get_param('ardrone_driver/control_yaw', 1.75)  # rads/s

    cmd_rate       = rospy.get_param('cmd_rate', 70);         # command rate (Hz)
    self.COMMAND_PERIOD = 1.0/cmd_rate

    # Design Parameters
    print "Getting parameters" 
    self.tau_x = rospy.get_param("~tau_x", 0.7)
    self.tau_y = rospy.get_param("~tau_y", 0.7)
    self.tau_z = rospy.get_param("~tau_z", 0.7)
    self.tau_w = rospy.get_param("~tau_w", 1.5)
    self.zeta  = rospy.get_param("~zeta",0.707);
    self.L1_type = rospy.get_param("~L1_type",1)

    print self.L1_type

    ###########################################################################
    # L1 adaptive output control parameters -- for L1 control of x_dot
    ###########################################################################
    
    ### L1 low pass filter
#    self.omega_cutoff = np.diag( np.array( [1.4, 1.4, 1.5] ) ) # first order
    #self.omega_cutoff = np.diag( np.array( [1.85, 1.3, 6.5] ) ) # third order (om_cz=9.0 is too high)
    self.omega_cutoff = np.diag( np.array( [1.7, 1.7, 6.75] ) ) # third order
#    self.omega_cutoff = np.diag( np.array( [8.0, 8.0, 6.75] ) ) # low-level third order

    #self.K = 20*np.diag(np.array([6.0, 6.0, 8.0]))
    # self.K = np.diag(np.array([1.0, 1.0, 1.0]))
    self.K = np.diag(np.array([0.8, 0.8, 2.0]))

    self.x_L1_des = np.array([[0.0],[0.0],[0.0]])

    # 3rd order filter initialization
    self.u_dot = np.array([[0.0],[0.0],[0.0]])
    self.u = np.array([[0.0],[0.0],[0.0]])
    self.y_ddot = np.array([[0.0],[0.0],[0.0]])
    self.y_dot = np.array([[0.0],[0.0],[0.0]])
    self.y = np.array([[0.0],[0.0],[0.0]])

    self.oldtime = rospy.get_rostime() # for integration
    
    # proportional gain for outer position loop
    self.old_pos_error = np.array([[0.0],[0.0],[0.0]])
#    self.Pgain = np.array([[0.5],[0.5],[0.5]]) # first order
    #self.Pgain = np.array([[0.57],[0.2],[0.75]])
    self.Pgain = np.array([[0.25],[0.25],[1.00]]) # third order
#    self.Pgain = np.array([[0.45],[0.075],[1.00]]) # low level - third order - P only
#    self.Pgain = np.array([[0.9],[0.5],[1.00]]) # low level - third order - PD
    self.Dgain = 0*0.08*np.array([[0.6],[0.85],[0.0]]) # low level - third order
###
    ### L1 adaptive estimation ###
    self.Gamma = 80.0 # L1 adaptive gain (80 is good for z-direction) # third order
#    self.Gamma = np.array([[100.0], [100.0], [80.0]]) # L1 adaptive gain # low-level third order

    self.sigma_hat = np.array([[0.0],[0.0],[0.0]]) # adaptive estimate
    
    ### Projection Operator - convex set ###
    self.sigma_hat_max = 20.0 # maximum absolute nominal value of sigma_hat
    self.epsilon_sigma = 0.1 # tolerance on maximum sigma_hat

    self.desired_vel = np.array([[0.0],[0.0],[0.0]])

    ### Reference Model -- first-order reference model M(s) = m/(s+m)*eye(3) ###
    # M_i(s) = m_i/(s+m_i), i = x,y,z
    # A_m = diag(-mx -my -mz), B_m = diag(mx my mz)
    ##self.A_m = np.diag(np.array([-4.0, -1.0, -1.0]))
    #self.A_m = np.diag(np.array([-25.0, -25.0, -4.5])) # USE THIS FOR L1 OUTPUT POSITION
#    self.A_m = np.diag(np.array([-15.0, -15.0, -25.0])) # USE THIS FOR L1 OUTPUT VELOCITY >> first order C
    self.A_m = np.diag(np.array([-2.4, -2.4, -7.0])) # USE THIS FOR L1 OUTPUT VELOCITY >> third order C
#    self.A_m = np.diag(np.array([-15.0, -15.0, -7.0])) # L1 low level - third order C
    self.B_m = -self.A_m
    
    self.B_inv = np.linalg.inv(self.B_m)

    self.omega_0 = np.eye(3)
    
    self.x_ref = np.array([[0.0],[0.0],[0.0]]) # initialize reference x position

    self.old_err = np.array([[0.0],[0.0],[0.0]])

    # Create csv file
    with open('/home/dsl5/l1_ref_output.csv','ab') as ref_model:
      writer = csv.writer(ref_model)
      # time secs, time nsecs, x_ref(1:3), x_dot(1:3), sigma_hat(1:3), x_L1_des(1:3), x_dot_des(1:3), x(1:3), x_des(1:3), rpy(1:3)
      writer.writerow(np.array([now.secs, now.nsecs, self.x_ref[0][0], self.x_ref[1][0], self.x_ref[2][0], curr.x_dot[0], curr.x_dot[1], curr.x_dot[2], self.sigma_hat[0][0], self.sigma_hat[1][0], self.sigma_hat[2][0], self.x_L1_des[0][0], self.x_L1_des[1][0], self.x_L1_des[2][0], self.desired_vel[0][0], self.desired_vel[1][0], self.desired_vel[2][0], curr.x[0], curr.x[1], curr.x[2], des.x[0], des.x[1], des.x[2], filtered_track_error[0][0], filtered_track_error[1][0], filtered_track_error[2][0]]))

    print "created csv data file"

    ###########################################################################

    print 'tau_x: ', self.tau_x, '\n', 'tau_y: ', self.tau_y, '\n', 'tau_z: ', self.tau_z, '\n', 'tau_w: ', self.tau_w, '\n', 'zeta: ', self.zeta, '\n'
    print 'euler_angle_max: ', self.max_euler, '\n', 'control_vz_max: ', self.max_vz, '\n', 'control_yaw: ', self.max_yaw, '\n' 
    
    #print 'L1 Adaptive Output control parameters \n', 'Adaptive Gain:',self.Gamma,'\n', 'sigma_hat_max:',self.sigma_hat_max,'\n', 'epsilon_sigma:',self.epsilon_sigma,'\n', 'cutoff frequency:',self.omega_cutoff,'\n'

    # Requests path waypoint
    self.pub_request  = rospy.Publisher('waypoint_request', Bool)

    # Establish a timer to request waypoints at a given frequency
    self.waypointTimer  = rospy.Timer(rospy.Duration(self.COMMAND_PERIOD), self.requestWaypoint)

    # Signal Handling in case ROS is shut down (video display closed or <Ctrl-C>)
    rospy.on_shutdown(self.hook)
    rospy.loginfo('Nonlinear controller ready-----------------------------------')

  #*****************************************************************************

  # signal handeling in case of shutdown
  def hook(self):
    self.pubLand.publish(Empty())
    print "Landing!"

  #*****************************************************************************

  def requestWaypoint(self,event):
    self.pub_request.publish(self.status.request_waypoint)

  #*****************************************************************************
  def determineCommands(self):

    # Save variables so they are not over-written in mid-calculation
    des  = State.fromState(self.desired_state)
    curr = State.fromState(self.current_state)
    
    # calculate time since last determineCommand call for integration purposes
    now = rospy.get_rostime()
    dt = now.secs-self.oldtime.secs + (now.nsecs-self.oldtime.nsecs)*0.000000001
    if dt == 0:
      #print 'now: ', now.secs, ' ', now.nsecs*0.000000001, '\n', 'old: ', self.oldtime.secs, ' ', self.oldtime.nsecs*0.000000001, '\n dt: ', dt
      dt = 0.001

    # store old time for next call
    self.oldtime = now
    
    ###########################################################################
    # L1 Output Feedback Implementation
    #    The L1 adaptive output feedback structure revises the desired state 
    ###########################################################################
    
    if self.L1_type == 3:
      # then use Projection based l1 output feedback on pitch, roll and z velocity
      
      ##### NOTE #####
      ##### x_L1_des = [ [phi_L1_des], [theta_L1_des], [z_dot_L1_des] ]
      ##### x_ref = [ [phi_ref], [theta_ref], [z_dot_ref] ]
      ##### 
      
      # first check whether drone is in flying mode
      # only use L1 when flying: 2 - landed, 6 - taking off, 3 - flying
      if (self.status.drone_state != 3): #and (self.status.drone_state !=7):
        # use standard controller when not in flying mode
        
        #self.x_L1_des = np.reshape(des.x, (3,-1))
        #y_tilde = np.array([[0.0],[0.0],[0.0]])
        
        # Z-velocity command m/sec)
        z_velocity_out =  ((2.0*self.zeta/self.tau_z) * (des.x_dot[2] - curr.x_dot[2]) + (1.0/(self.tau_z**2))*(des.x[2] - curr.x[2]) )
    
        # calculate the desired acceleration in x and y (global coordinates, [m/s^2] )
        ax = (2.0*self.zeta/self.tau_x)*(des.x_dot[0] - curr.x_dot[0]) + (1.0/(self.tau_x*self.tau_x))*(des.x[0]-curr.x[0])
        ay = (2.0*self.zeta/self.tau_x)*(des.x_dot[1] - curr.x_dot[1]) + (1.0/(self.tau_x*self.tau_x))*(des.x[1]-curr.x[1])
  
        # Roll/Pitch Commands
        # determine the mass-normalized thrust
        thrust = np.linalg.norm(np.array([0.,0.,self.g]) + curr.x_ddot)
  
        # keep ax,ay < thrust (so that arcsin is defined)
        if thrust == 0.0:
          print "########## thrust is zero ##########"
          ax_clamped = 1.0
          ay_clamped = 1.0
        else:
          ax_clamped = self.clamp(ax / thrust, 1.0)
          ay_clamped = self.clamp(ay / thrust, 1.0)
     
        # Rotate desired accelerations into drone's body frame
        ax_b =  ax_clamped*np.cos(curr.rpy[2]) + ay_clamped*np.sin(curr.rpy[2])
        ay_b = -ax_clamped*np.sin(curr.rpy[2]) + ay_clamped*np.cos(curr.rpy[2])
    
        ax_b = self.clamp(ax_b, 0.95)
        ay_b = self.clamp(ay_b, 0.95)

        # convert acceleration into roll/pitch angles [rad]
        pitch_out =  np.arcsin(ax_b)
        roll_out  = -np.arcsin(ay_b)
        
        
      else:
        # use L1 control when in flying mode
        # calculate error between actual and reference state position
        y_tilde = self.x_ref - np.array( [ [curr.rpy[0]], [curr.rpy[1]], [curr.x_dot[2]] ] )
        
        ### Projection Operator to update sigma_hat based on y_tilde            ###
        f = ((self.epsilon_sigma + 1.0)*(self.sigma_hat.T.dot( self.sigma_hat )[0][0] ) - self.sigma_hat_max**2)/(self.epsilon_sigma*self.sigma_hat_max**2)
        grad_f = 2.0*(self.epsilon_sigma + 1.0)/(self.epsilon_sigma*self.sigma_hat_max**2)*self.sigma_hat
        
        if f<0:
          projection_result = -y_tilde
        else:
          if -grad_f.T.dot(y_tilde)[0][0] <0:
            projection_result = -y_tilde
          else:
            projection_result = -y_tilde + (1/np.linalg.norm(grad_f))*(grad_f)*grad_f.T.dot(y_tilde)[0][0]*f
            
        # multiply by adaptive Gain and integrate 
        sigma = self.sigma_hat + dt*(self.Gamma*projection_result)
        
        sigma_x = self.clamp(sigma[0][0], self.sigma_hat_max*(1+self.epsilon_sigma) )
        sigma_y = self.clamp(sigma[1][0], self.sigma_hat_max*(1+self.epsilon_sigma) )
        sigma_z = self.clamp(sigma[2][0], self.sigma_hat_max*(1+self.epsilon_sigma) )
        self.sigma_hat = np.array([[sigma_x],[sigma_y],[sigma_z]])
        #self.sigma_hat = self.clamp(self.sigma_hat + dt*(-self.Gamma*y_tilde), self.sigma_hat_max)
          
        # find desired velocity for L1 output from proportional-d controller
        pos_error = np.reshape(des.x, (3,-1)) - np.reshape(curr.x, (3,-1))
        self.desired_vel = self.Pgain*( pos_error ) + self.Dgain*(1/dt)*(self.old_pos_error - pos_error)
        self.old_pos_error = pos_error
       
        ### Find revised x_dot desired by low-pass filtering tracking error ###
        #track_error = np.reshape(des.x_dot, (3,-1)) - self.sigma_hat
        track_error = self.desired_vel - self.sigma_hat  
        
###	Third Order Low Pass Filter y = C(s)*u
        # low pass filter C(s) = (3*omega_cutoff^2*s + omega_cutoff^3)/(s^3 + 3*omega_cutoff*s^2 + 3*omega_cutoff^2*s + omega_cutoff^3)
        
        # first find derivative of input signal (i.e. u = track_error, u_dot = d/dt(track_error) )
        self.u_dot = 1/dt*(track_error - self.u) # u_dot = 1/dt*(u - u_old)
        self.u = track_error # set current u to track_error (in next iteration, this is automatically u_old)
        
        self.y_ddot = self.y_ddot + dt*(-3*self.omega_cutoff.dot(self.y_ddot) - 3*(self.omega_cutoff**2).dot(self.y_dot) - (self.omega_cutoff**3).dot(self.y) + 3*(self.omega_cutoff**2).dot(self.u_dot) + (self.omega_cutoff**3).dot(self.u) )
        self.y_dot = self.y_dot + dt*(self.y_ddot)
        self.y = self.y + dt*(self.y_dot)
        
        # low pass filter output is L1 desired
        filtered_track_error = self.y

        # z-translational velocity is unchanged (i.e. directly passed from low pass filter output to ARDrone)
        self.x_L1_des[2][0] = filtered_track_error[2][0]
        
        ### translate output of the low pass filter (i.e. x and y directions) into I-frame roll and pitch angles, and tranform these according to yaw angle to B-frame roll and pitch angles
        
        # determine the mass-normalized thrust
        thrust = np.linalg.norm(np.array([0.,0.,self.g]) + curr.x_ddot)
        
        # keep ax,ay < thrust (so that arcsin is defined)
        if thrust == 0.0:
          print "########## thrust is zero ##########"
          ax_clamped = 1.0
          ay_clamped = 1.0
        else:
          ax_clamped = self.clamp(filtered_track_error[0][0] / thrust, 1.0)
          ay_clamped = self.clamp(filtered_track_error[1][0] / thrust, 1.0)
     
        # Rotate desired accelerations into drone's body frame
        ax_b =  ax_clamped*np.cos(curr.rpy[2]) + ay_clamped*np.sin(curr.rpy[2])
        ay_b = -ax_clamped*np.sin(curr.rpy[2]) + ay_clamped*np.cos(curr.rpy[2])
     
        ax_b = self.clamp(ax_b, 0.95)
        ay_b = self.clamp(ay_b, 0.95)

        # convert acceleration into roll/pitch angles [rad]
        self.x_L1_des[1][0] =  np.arcsin(ax_b) # pitch
        self.x_L1_des[0][0]  = -np.arcsin(ay_b) # roll



        ### reference model -- M(s) = m/(s+m) -- x_ref = M(s)(u + sigma_hat) ###
        self.x_ref = self.x_ref + dt*self.B_m.dot( -self.x_ref + self.x_L1_des + self.sigma_hat )
        
        # append to csv file
        with open('/home/dsl5/l1_ref_output.csv','ab') as ref_model:
          writer = csv.writer(ref_model)
          # time secs, time nsecs, x_ref(1:3), x_dot(1:3), sigma_hat(1:3), x_L1_des(1:3), x_dot_des(1:3), x(1:3), x_des(1:3), rpy(1:3)
          writer.writerow(np.array([now.secs, now.nsecs, self.x_ref[0][0], self.x_ref[1][0], self.x_ref[2][0], curr.x_dot[0], curr.x_dot[1], curr.x_dot[2], self.sigma_hat[0][0], self.sigma_hat[1][0], self.sigma_hat[2][0], self.x_L1_des[0][0], self.x_L1_des[1][0], self.x_L1_des[2][0], self.desired_vel[0][0], self.desired_vel[1][0], self.desired_vel[2][0], curr.x[0], curr.x[1], curr.x[2], des.x[0], des.x[1], des.x[2], filtered_track_error[0][0], filtered_track_error[1][0], filtered_track_error[2][0]]))
          
#        self.pubL1des_x.publish(self.x_L1_des[0][0])
#        self.pubL1des_y.publish(self.x_L1_des[1][0])
#        self.pubL1des_z.publish(self.x_L1_des[2][0])
#  
#        self.pubParama.publish(self.sigma_hat[0][0])
#        self.pubParamb.publish(self.sigma_hat[1][0])
#        self.pubParamc.publish(self.sigma_hat[2][0])
#    
#        self.pubXref_x.publish(self.x_ref[0][0])
#        self.pubXref_y.publish(self.x_ref[1][0])
#        self.pubXref_z.publish(self.desired_vel[2][0])
      
      ###########################################################################

        z_velocity_out = self.x_L1_des[2][0]
        pitch_out =  self.x_L1_des[1][0]
        roll_out  = self.x_L1_des[0][0]



      # Yaw rate command (rad/sec)??
      yaw_err = np.mod(des.rpy[2]-curr.rpy[2] + np.pi, 2.*np.pi) - np.pi
      yaw_velocity_out = (1.0 / self.tau_w) * yaw_err

      # check for valid outputs and limit if necessary
      if pitch_out > 0.75:
        print "pitch: ", pitch_out
        print "ax: ", ax
        print "ay: ", ay
        print "dt: ", dt
        pitch_out = 0.0
      elif pitch_out < -0.75:
        print "roll: ", pitch_out
        print "ax: ", ax
        print "ay: ", ay
        print "dt: ", 
        pitch_out = -0.75
      elif np.isnan(pitch_out):
        print "*******************ax_b: ", ax_b
        pitch_out = 0.0
      
      if roll_out > 0.75:
        print "roll: ", roll_out
        print "ax: ", ax
        print "ay: ", ay
        roll_out = 0.75
      elif roll_out < -0.75:
        print "roll: ", roll_out
        print "ax: ", ax
        print "ay: ", ay
        roll_out = -0.75
      elif np.isnan(roll_out):
        print "*******************ay_b: ", ay_b
        roll_out = 0.0
      
#    self.pubXcurr_x.publish(curr.x[0])
#    self.pubXcurr_y.publish(curr.x[1])
#    self.pubXcurr_z.publish(curr.x[2])
#
#    self.pubXdotcurr_x.publish(curr.x_dot[0])
#    self.pubXdotcurr_y.publish(curr.x_dot[1])
#    self.pubXdotcurr_z.publish(curr.x_dot[2])


#############
# DEBUGGING #
#############
#      self.command.twist.angular.x = (des.x[0]-curr.x[0]);
#      self.command.twist.angular.y = (des.x_dot[0]-curr.x_dot[0]);
    

      #### send the commands to the drone if the keyboard is not currently being used
      if(self.status.keyboard_override == 0):
        self.status.t_last_cmd = time.time()
  
        if math.isnan(pitch_out):
          print "pitch is NaN before sendCommand"

        if math.isnan(roll_out):
          print "roll is NaN before sendCommand"
  
      
        with open('/home/dsl5/l1_angles.csv','ab') as angles:
          writer = csv.writer(angles)
          writer.writerow(np.array([roll_out, pitch_out, yaw_velocity_out, z_velocity_out, curr.rpy[0], curr.rpy[1], curr.x_dot[2], now.secs, now.nsecs]))
        
        self.SendCommand(roll_out, pitch_out, yaw_velocity_out, z_velocity_out)

###############################################################################

    else:
      #print "starting higher level L1 output feedback controller"
      # use higher level L1 controllers
      
      if self.L1_type == 1:
        # then use Piecewise Constant l1 output feedback on position
        # first check whether drone is in flying mode
        
        # only use L1 when flying: 2 - landed, 6 - taking off, 3 - flying
        if self.status.drone_state != 3:
          self.x_L1_des = np.reshape(des.x, (3,-1))
          y_tilde = np.array([[0.0],[0.0],[0.0]])
          #print self.x_L1_des
          #print 'no L1, start flying'
        else:
          # use L1 adaptive output
          
          # calculate error between actual and reference state position
          y_tilde = self.x_ref - np.reshape(curr.x, (3,-1))
          #self.pubYtilde_x.publish(self.y_tilde[0][0])
          #self.pubYtilde_y.publish(self.y_tilde[1][0])
          #self.pubYtilde_z.publish(self.y_tilde[2][0])
        
          # ensure dt is not 0
          if dt <= 0.0001:
            #print 'dt min reached: ', dt
            dt = 0.001
            
          #m = 20.0
          #k = 0.5
#          m = 25.0
#          k = 0.8
          
          self.sigma_hat[0][0] = -1.0/self.B_m[0][0] * ( 1.0 / (math.exp(self.A_m[0][0]*dt) - 1.0) ) * (1.0/self.A_m[0][0]) * math.exp(self.A_m[0][0]*dt) * y_tilde[0][0]
          self.sigma_hat[1][0] = -1.0/self.B_m[1][1] * ( 1.0 / (math.exp(self.A_m[1][1]*dt) - 1.0) ) * (1.0/self.A_m[1][1]) * math.exp(self.A_m[1][1]*dt) * y_tilde[1][0]
          self.sigma_hat[2][0] = -1.0/self.B_m[2][2] * ( 1.0 / (math.exp(self.A_m[2][2]*dt) - 1.0) ) * (1.0/self.A_m[2][2]) * math.exp(self.A_m[2][2]*dt) * y_tilde[2][0]
  
          
#          eta_hat = 1.0*self.x_L1_des[0][0] + self.sigma_hat[0][0] - des.x[0]
#  
#          self.x_L1_des[0][0] = self.x_L1_des[0][0] + dt*(-k*eta_hat)
#          
#          self.x_ref[0][0] = self.x_ref[0][0] + dt*(m)*( -self.x_ref[0][0] + self.x_L1_des[0][0] + self.sigma_hat[0][0] )
          
          
#          ### Piecewise constant update of sigma_hat based on y_tilde ###
#          ### NOTE: Implementation currently assumes A_m diagonal
#          # calculate matrix exponential of A_m*T_s
#          Am_exp = np.diag(np.array([math.exp( self.A_m[0][0] * dt ), math.exp( self.A_m[1][1] * dt ), math.exp( self.A_m[2][2] * dt )]))
#          #print Am_exp[0][0] - 1, Am_exp[1][1] - 1, Am_exp[2][2] - 1
#          #print 'dt: ', dt
#          
#          # calculate Phi(T_s)
#          Phi_inv = np.diag(np.array( [ 1.0/(Am_exp[0][0] - 1.0), 1.0/(Am_exp[1][1] - 1.0), 1.0/(Am_exp[2][2] - 1.0) ] ))
#          #if np.isnan(1/(Am_exp[0][0] - 1)) or np.isinf(1/(Am_exp[0][0] - 1)):
#          #  print 1/(Am_exp[0][0] - 1)
#          # Phi_inv = np.linalg.inv(Am_exp - np.eye(3))
#          Phi_inv = Phi_inv.dot( self.A_m )
#          #print 'Phi_inv', '\n', Phi_inv, '\n' 
#        
#          # calculate sigma_hat
#          self.sigma_hat = -self.B_inv.dot( Phi_inv.dot( Am_exp.dot( y_tilde ) ) )
#          
#          #self.sigma_hat = np.array( [ [-self.B_inv[0][0]*Phi_inv[0][0]*Am_exp[0][0]*y_tilde[0][0]] , [-self.B_inv[1][1]*Phi_inv[1][1]*Am_exp[1][1]*y_tilde[1][0]] , [-self.B_inv[2][2]*Phi_inv[2][2]*Am_exp[2][2]*y_tilde[2][0]] ] )
#    
#          #sigma_x = self.clamp(self.sigma_hat[0][0],30)
#          #if np.isnan(sigma_x):
#          #  sigma_x = 0
#          #sigma_y = self.clamp(self.sigma_hat[1][0],30)
#          #if np.isnan(sigma_y):
#          #  sigma_y = 0
#          #sigma_z = self.clamp(self.sigma_hat[2][0],30)
#          #if np.isnan(sigma_z):
#          #  sigma_z = 0
#          #self.sigma_hat = np.array([[sigma_x],[sigma_y],[sigma_z]])
#          
#          #if np.isnan(self.sigma_hat[0][0]):
#          #  print 'sigma_hat', '\n', self.sigma_hat, '\n'
        
        
          ### Find revised x desired L1 ###
          
          # exact tracking: r_g = K_g * r, K_g = -inv(C*inv(Am)*Bm)
          # C = eye(3), diag Am = -Bm => K_g = eye(3)
          # -np.linalg.inv( np.eye(3).dot( (np.linalg.inv(A_m)).dot(B_m) ) )
          # r_g(t) = r(t)
          r_g = np.reshape(des.x, (3,-1))
          
          # calculate intermediate signal for adaptive ouput controller
          eta_hat = self.omega_0.dot( self.x_L1_des ) + self.sigma_hat - r_g
          #print 'eta_hat', '\n', eta_hat
          
          # calculate revised x position -- D(s) = 1/s, simple integrator
          self.x_L1_des = self.x_L1_des + dt*( -self.K.dot( eta_hat ) )
          #print self.x_L1_des
          
          
          ### reference model -- x_dot_ref = Am*x_ref + Bm(omega_0*x_L1_des + sigma_hat) ###
          self.x_ref = self.x_ref + dt*self.B_m.dot( -self.x_ref + self.x_L1_des + self.sigma_hat )
          
          
        #print self.sigma_hat
        self.pubL1des_x.publish(self.x_L1_des[0][0])
        self.pubL1des_y.publish(self.x_L1_des[1][0])
        self.pubL1des_z.publish(self.x_L1_des[2][0])
    
        self.pubParama.publish(self.sigma_hat[0][0])
        self.pubParamb.publish(self.sigma_hat[1][0])
        self.pubParamc.publish(self.sigma_hat[2][0])
    
        self.pubXref_x.publish(self.x_ref[0][0])
        self.pubXref_y.publish(self.x_ref[1][0])
        self.pubXref_z.publish(self.x_ref[2][0])

        ###########################################################################

        # Z-velocity command m/sec)
        #z_velocity_out =  ((2.0*self.zeta/self.tau_z) * (des.x_dot[2] - curr.x_dot[2]) + (1.0/(self.tau_z**2))*(des.x[2] - curr.x[2]) )
        z_velocity_out = (1.0/(self.tau_z**2))*(self.x_L1_des[2][0] - curr.x[2]) ### NOTE: x_L1_des z-position
    
        # calculate the desired acceleration in x and y (global coordinates, [m/s^2] )
        #ax = (2.0*self.zeta/self.tau_x)*(0 - curr.x_dot[0]) + (1.0/(self.tau_x*self.tau_x))*(self.x_L1_des[0][0]-curr.x[0]) ### NOTE: x_L1_des x-position
        #ay = (2.0*self.zeta/self.tau_x)*(0 - curr.x_dot[1]) + (1.0/(self.tau_x*self.tau_x))*(self.x_L1_des[1][0]-curr.x[1]) ### NOTE: x_L1_des y-position
        #ax = (1.0/(self.tau_x*self.tau_x))*0.1*(self.x_L1_des[0][0]-curr.x[0]) ### NOTE: x_L1_des x-position
        #ax = (1.0/(self.tau_x*self.tau_x))*(des.x[0]-curr.x[0])
        #ay = (2.0*self.zeta/self.tau_x)*0.1*(des.x_dot[1] - curr.x_dot[1]) + (1.0/(self.tau_x*self.tau_x))*(des.x[1]-curr.x[1]) ### NOTE: x_L1_des y-position
  
        ax = (2.0*self.zeta/self.tau_x)*0.35*(1.0/dt)*(self.x_L1_des[0][0] - curr.x[0] - self.old_err[0][0]) + (1.0/(self.tau_x*self.tau_x))*1.5*(self.x_L1_des[0][0]-curr.x[0]) ### NOTE: x_L1_des x-position
        ay = (2.0*self.zeta/self.tau_x)*0.35*(1.0/dt)*(self.x_L1_des[1][0] - curr.x[1] - self.old_err[1][0]) + (1.0/(self.tau_x*self.tau_x))*1.5*(self.x_L1_des[1][0]-curr.x[1]) ### NOTE: x_L1_des y-position
        self.old_err = self.x_L1_des - np.reshape(curr.x, (3,-1))

        #ay = (2.0*self.zeta/self.tau_x)*(des.x_dot[1] - curr.x_dot[1]) + (1.0/(self.tau_x*self.tau_x))*(self.x_L1_des[1][0]-curr.x[1]) ### NOTE: x_L1_des y-position
        #ax = (1.0/(self.tau_x*self.tau_x))*(self.x_L1_des[0][0]-curr.x[0]) ### NOTE: x_L1_des x-position
        #ay = (1.0/(self.tau_x*self.tau_x))*(self.x_L1_des[1][0]-curr.x[1]) ### NOTE: x_L1_des y-position

    
      ##### PROJECTION BASED L1 OUTPUT FEEDBACK on X-DOT #####
      
      elif self.L1_type == 2:
        # then use Projection based l1 output feedback on translational velocity
        # first check whether drone is in flying mode
  
        # only use L1 when flying: 2 - landed, 6 - taking off, 3 - flying
        if (self.status.drone_state != 3): #and (self.status.drone_state !=7):
    #    if 1:
          self.x_L1_des = np.reshape(des.x, (3,-1))
          y_tilde = np.array([[0.0],[0.0],[0.0]])
          #print self.x_L1_des
          #print 'no L1, start flying'
          #print self.status.drone_state
        else:
  
          # calculate error between actual and reference state position
          #y_tilde = self.x_ref - self.desired_vel
          y_tilde = self.x_ref - np.reshape(curr.x_dot, (3,-1))
          
          ### Projection Operator to update sigma_hat based on y_tilde            ###
          f = ((self.epsilon_sigma + 1.0)*(self.sigma_hat.T.dot( self.sigma_hat )[0][0] ) - self.sigma_hat_max**2)/(self.epsilon_sigma*self.sigma_hat_max**2)
          grad_f = 2.0*(self.epsilon_sigma + 1.0)/(self.epsilon_sigma*self.sigma_hat_max**2)*self.sigma_hat
      
          if f<0:
            projection_result = -y_tilde
          else:
            if -grad_f.T.dot(y_tilde)[0][0] <0:
              projection_result = -y_tilde
            else:
              projection_result = -y_tilde + (1/np.linalg.norm(grad_f))*(grad_f)*grad_f.T.dot(y_tilde)[0][0]*f
        
          # multiply by adaptive Gain and integrate 
          sigma = self.sigma_hat + dt*(self.Gamma*projection_result)
          
          sigma_x = self.clamp(sigma[0][0], self.sigma_hat_max*(1+self.epsilon_sigma) )
          sigma_y = self.clamp(sigma[1][0], self.sigma_hat_max*(1+self.epsilon_sigma) )
          sigma_z = self.clamp(sigma[2][0], self.sigma_hat_max*(1+self.epsilon_sigma) )
          self.sigma_hat = np.array([[sigma_x],[sigma_y],[sigma_z]])
          #self.sigma_hat = self.clamp(self.sigma_hat + dt*(-self.Gamma*y_tilde), self.sigma_hat_max)
          #print self.sigma_hat
          
#          f_x = ((self.epsilon_sigma + 1.0)*(self.sigma_hat[0][0] * self.sigma_hat[0][0] ) - self.sigma_hat_max**2)/(self.epsilon_sigma*self.sigma_hat_max**2)
#          f_y = ((self.epsilon_sigma + 1.0)*(self.sigma_hat[1][0] * self.sigma_hat[1][0] ) - self.sigma_hat_max**2)/(self.epsilon_sigma*self.sigma_hat_max**2)
#          f_z = ((self.epsilon_sigma + 1.0)*(self.sigma_hat[2][0] * self.sigma_hat[2][0] ) - self.sigma_hat_max**2)/(self.epsilon_sigma*self.sigma_hat_max**2)
#          grad_f = 2.0*(self.epsilon_sigma + 1.0)/(self.epsilon_sigma*self.sigma_hat_max**2)*self.sigma_hat
#    
#          if f_x < 0:
  #          projection_result_x = -y_tilde[0][0]
#          else:
#            if -grad_f[0][0] * y_tilde[0][0] < 0:
#              projection_result_x = -y_tilde[0][0]
#            else:
#              projection_result_x = -y_tilde[0][0] + grad_f[0][0]*y_tilde[0][0]*f_x
#
#          if f_y < 0:
#            projection_result_y = -y_tilde[1][0]
#          else:
#            if -grad_f[1][0] * y_tilde[1][0] < 0:
#              projection_result_y = -y_tilde[1][0]
#            else:
#              projection_result_y = -y_tilde[1][0] + grad_f[1][0]*y_tilde[1][0]*f_y
#
#          if f_z < 0:
#            projection_result_z = -y_tilde[2][0]
#          else:
#            if -grad_f[2][0] * y_tilde[2][0] < 0:
#              projection_result_z = -y_tilde[2][0]
#            else:
#              projection_result_z = -y_tilde[2][0] + grad_f[2][0]*y_tilde[2][0]*f_z
#        
#          ## multiply by adaptive Gain and integrate 
#          #sigma = self.sigma_hat[0][0] + dt*(self.Gamma*projection_result)
#          
#          sigma_x = self.clamp( self.sigma_hat[0][0] + dt*(self.Gamma*projection_result_x) , self.sigma_hat_max*(1+self.epsilon_sigma) )
#          sigma_y = self.clamp( self.sigma_hat[1][0] + dt*(self.Gamma*projection_result_y) , self.sigma_hat_max*(1+self.epsilon_sigma) )
#          sigma_z = self.clamp( self.sigma_hat[2][0] + dt*(self.Gamma*projection_result_z) , self.sigma_hat_max*(1+self.epsilon_sigma) )
#          self.sigma_hat = np.array([[sigma_x],[sigma_y],[sigma_z]])
#          #self.sigma_hat = self.clamp(self.sigma_hat + dt*(-self.Gamma*y_tilde), self.sigma_hat_max)
#          #if not np.isnan(self.sigma_hat[0][0]):
#          #  print 'sigma_hat', '\n', self.sigma_hat, '\n'

                  
          # find desired velocity for L1 output from proportional controller
          #desired_vel = 1.0*( -np.reshape(des.x, (3,-1)) + np.reshape(curr.x, (3,-1)) )
          #self.desired_vel = 0.3*(self.desired_vel + dt*1.0*( -np.reshape(des.x, (3,-1)) + np.reshape(curr.x, (3,-1)) ))
          self.desired_vel = self.Pgain*( np.reshape(des.x, (3,-1)) - np.reshape(curr.x, (3,-1)) )

        
          ### Find revised x_dot desired by low-pass filtering tracking error ###
          #track_error = np.reshape(des.x_dot, (3,-1)) - self.sigma_hat
          track_error = self.desired_vel - self.sigma_hat
        
        
###	First Order Low Pass Filter
          # low pass filter C(s) = omega_cutoff / (s + omega_cutoff) ### NOTE: decoupled directions
#          self.x_L1_des = self.x_L1_des + dt*self.omega_cutoff.dot( -self.x_L1_des + track_error )
##          self.x_L1_des[0][0] = self.x_L1_des[0][0] + dt*1.5*( -self.x_L1_des[0][0] + track_error[0][0] )
##          self.x_L1_des[1][0] = self.x_L1_des[1][0] + dt*1.5*( -self.x_L1_des[1][0] + track_error[1][0] )
##          self.x_L1_des[2][0] = self.x_L1_des[2][0] + dt*2.0*( -self.x_L1_des[2][0] + track_error[2][0] )
        
        
###	Third Order Low Pass Filter y = C(s)*u
#          # low pass filter C(s) = (3*omega_cutoff^2*s + omega_cutoff^3)/(s^3 + 3*omega_cutoff*s^2 + 3*omega_cutoff^2*s + omega_cutoff^3)
#          
#          # first find derivative of input signal (i.e. u = track_error, u_dot = d/dt(track_error) )
          self.u_dot = 1/dt*(track_error - self.u) # u_dot = 1/dt*(u - u_old)
          self.u = track_error # set current u to track_error (in next iteration, this is automatically u_old)
        
          self.y_ddot = self.y_ddot + dt*(-3*self.omega_cutoff.dot(self.y_ddot) - 3*(self.omega_cutoff**2).dot(self.y_dot) - (self.omega_cutoff**3).dot(self.y) + 3*(self.omega_cutoff**2).dot(self.u_dot) + (self.omega_cutoff**3).dot(self.u) )
          self.y_dot = self.y_dot + dt*(self.y_ddot)
          self.y = self.y + dt*(self.y_dot)
        
          # low filter output is L1 desired velocity
          self.x_L1_des = self.y
        
#          print self.x_dot_L1_des

#          if math.isnan(self.x_L1_des[0][0]):
#            print "L1 des x is nan"
#            self.x_L1_des[0][0] = 0.0
#
#          if math.isnan(self.x_L1_des[1][0]):
#            print "L1 des y is nan"
#            self.x_L1_des[1][0] = 0.0
#
#          if math.isnan(self.x_L1_des[2][0]):
#            print "L1 des z is nan"
#            self.x_L1_des[2][0] = 0.0

          ### reference model -- M(s) = m/(s+m) -- x_ref = M(s)(u + sigma_hat) ###
          self.x_ref = self.x_ref + dt*self.B_m.dot( -self.x_ref + self.x_L1_des + self.sigma_hat )
#          self.x_ref[0][0] = self.x_ref[0][0] + dt*15.0*( -self.x_ref[0][0] + self.x_L1_des[0][0] + self.sigma_hat[0][0] )
#          self.x_ref[1][0] = self.x_ref[1][0] + dt*15.0*( -self.x_ref[1][0] + self.x_L1_des[1][0] + self.sigma_hat[1][0] )
#          self.x_ref[2][0] = self.x_ref[2][0] + dt*20.0*( -self.x_ref[2][0] + self.x_L1_des[2][0] + self.sigma_hat[2][0] )
          #self.x_dot_ref = self.x_dot_ref + dt*self.m*(-self.x_dot_ref + self.x_dot_L1_des + self.sigma_hat)
        
          # append to csv file
          with open('/home/dsl5/l1_ref_output.csv','ab') as ref_model:
            writer = csv.writer(ref_model)
            # time secs, time nsecs, x_ref(1:3), x_dot(1:3), sigma_hat(1:3), x_L1_des(1:3), x_dot_des(1:3), x(1:3), x_des(1:3)
            writer.writerow(np.array([now.secs, now.nsecs, self.x_ref[0][0], self.x_ref[1][0], self.x_ref[2][0], curr.x_dot[0], curr.x_dot[1], curr.x_dot[2], self.sigma_hat[0][0], self.sigma_hat[1][0], self.sigma_hat[2][0], self.x_L1_des[0][0], self.x_L1_des[1][0], self.x_L1_des[2][0], self.desired_vel[0][0], self.desired_vel[1][0], self.desired_vel[2][0], curr.x[0], curr.x[1], curr.x[2], des.x[0], des.x[1], des.x[2]]))
  
          #if not np.isnan(self.sigma_hat[0][0]):
            #print 'y-tilde', '\n', y_tilde, '\n'
            #print 'sigma_hat', '\n', self.sigma_hat, '\n'
            #print 'des_vel', '\n', self.desired_vel, '\n'
            #print 'x_dot_l1_des', '\n', self.x_L1_des, '\n'
          
          self.pubL1des_x.publish(self.x_L1_des[0][0])
          self.pubL1des_y.publish(self.x_L1_des[1][0])
          self.pubL1des_z.publish(self.x_L1_des[2][0])
    
          self.pubParama.publish(self.sigma_hat[0][0])
          self.pubParamb.publish(self.sigma_hat[1][0])
          self.pubParamc.publish(self.sigma_hat[2][0])
      
          self.pubXref_x.publish(self.x_ref[0][0])
          self.pubXref_y.publish(self.x_ref[1][0])
          self.pubXref_z.publish(self.desired_vel[2][0])
      
      ###########################################################################

        # Z-velocity command m/sec)
        # z_velocity_out =  ((2.0*self.zeta/self.tau_z) * (des.x_dot[2] - curr.x_dot[2]) + (1.0/(self.tau_z**2))*(des.x[2] - curr.x[2]) )
        #z_velocity_out = (1.0/(self.tau_z**2))*(self.x_L1_des[2][0] - curr.x[2]) ### NOTE: x_L1_des z-position
        #z_velocity_out = (1.0/(self.tau_z**2))*(des.x[2] - curr.x[2])
        z_velocity_out = self.x_L1_des[2][0]
  
    
        # calculate the desired acceleration in x and y (global coordinates, [m/s^2] )
        ax = (2.0*self.zeta/self.tau_x)*1.0*(self.x_L1_des[0][0] - curr.x_dot[0]) ### NOTE: x_L1_des x-velocity
        ay = (2.0*self.zeta/self.tau_x)*1.0*(self.x_L1_des[1][0] - curr.x_dot[1]) ### NOTE: x_L1_des y-velocity
        #ay = (2.0*self.zeta/self.tau_x)*(des.x_dot[1] - curr.x_dot[1]) + (1.0/(self.tau_x*self.tau_x))*(des.x[1]-curr.x[1])
        #ax = (2.0*self.zeta/self.tau_x)*(self.x_L1_des[0][0] - curr.x_dot[0]) + (1.0/(self.tau_x*self.tau_x))*(des.x[0]-curr.x[0]) ### NOTE: x_L1_des x-position
        #ay = (2.0*self.zeta/self.tau_x)*(self.x_L1_des[1][0] - curr.x_dot[1]) + (1.0/(self.tau_x*self.tau_x))*(des.x[0]-curr.x[1]) ### NOTE: x_L1_des y-position
        
      else:
        # use standard controller
        
        # Z-velocity command m/sec)
        z_velocity_out =  ((2.0*self.zeta/self.tau_z) * (des.x_dot[2] - curr.x_dot[2]) + (1.0/(self.tau_z**2))*(des.x[2] - curr.x[2]) )
    
        # calculate the desired acceleration in x and y (global coordinates, [m/s^2] )
        ax = (2.0*self.zeta/self.tau_x)*(des.x_dot[0] - curr.x_dot[0]) + (1.0/(self.tau_x*self.tau_x))*(des.x[0]-curr.x[0])
        ay = (2.0*self.zeta/self.tau_x)*(des.x_dot[1] - curr.x_dot[1]) + (1.0/(self.tau_x*self.tau_x))*(des.x[1]-curr.x[1])
      
      
      self.pubXcurr_x.publish(curr.x[0])
      self.pubXcurr_y.publish(curr.x[1])
      self.pubXcurr_z.publish(curr.x[2])

      self.pubXdotcurr_x.publish(curr.x_dot[0])
      self.pubXdotcurr_y.publish(curr.x_dot[1])
      self.pubXdotcurr_z.publish(curr.x_dot[2])

      # Yaw rate command (rad/sec)??
      yaw_err = np.mod(des.rpy[2]-curr.rpy[2] + np.pi, 2.*np.pi) - np.pi
      yaw_velocity_out = (1.0 / self.tau_w) * yaw_err
  
      # Roll/Pitch Commands
      # determine the mass-normalized thrust
      thrust = np.linalg.norm(np.array([0.,0.,self.g]) + curr.x_ddot)

      # keep ax,ay < thrust (so that arcsin is defined)
      if thrust == 0.0:
        print "########## thrust is zero ##########"
        ax_clamped = 1.0
        ay_clamped = 1.0
      else:
        ax_clamped = self.clamp(ax / thrust, 1.0)
        ay_clamped = self.clamp(ay / thrust, 1.0)
   
      # Rotate desired accelerations into drone's body frame
      ax_b =  ax_clamped*np.cos(curr.rpy[2]) + ay_clamped*np.sin(curr.rpy[2])
      ay_b = -ax_clamped*np.sin(curr.rpy[2]) + ay_clamped*np.cos(curr.rpy[2])
  
      ax_b = self.clamp(ax_b, 0.95)
      ay_b = self.clamp(ay_b, 0.95)

      # convert acceleration into roll/pitch angles [rad]
      pitch_out =  np.arcsin(ax_b)
      roll_out  = -np.arcsin(ay_b)
  
      if pitch_out > 0.75:
        print "pitch: ", pitch_out
        print "ax: ", ax
        print "ay: ", ay
        print "dt: ", dt
        pitch_out = 0.0
      elif pitch_out < -0.75:
        print "roll: ", pitch_out
        print "ax: ", ax
        print "ay: ", ay
        print "dt: ", 
        pitch_out = -0.75
      elif np.isnan(pitch_out):
        print "*******************ax_b: ", ax_b
        pitch_out = 0.0
  
      if roll_out > 0.75:
        print "roll: ", roll_out
        print "ax: ", ax
        print "ay: ", ay
        roll_out = 0.75
      elif roll_out < -0.75:
        print "roll: ", roll_out
        print "ax: ", ax
        print "ay: ", ay
        roll_out = -0.75
      elif np.isnan(roll_out):
        print "*******************ay_b: ", ay_b
        roll_out = 0.0
  
#############
# DEBUGGING #
#############
      self.command.twist.angular.x = (des.x[0]-curr.x[0]);
      self.command.twist.angular.y = (des.x_dot[0]-curr.x_dot[0]);
  
  
      # send the commands to the drone if the keyboard is not currently being used
      if(self.status.keyboard_override == 0):
        self.status.t_last_cmd = time.time()
  
        if math.isnan(pitch_out):
          print "pitch is NaN before sendCommand"
  
        if math.isnan(roll_out):
          print "roll is NaN before sendCommand"
  
      
        with open('/home/dsl5/l1_angles.csv','ab') as angles:
          writer = csv.writer(angles)
          writer.writerow(np.array([roll_out, pitch_out, yaw_velocity_out, z_velocity_out, curr.rpy[0], curr.rpy[1], curr.x_dot[2], now.secs, now.nsecs]))
        
        self.SendCommand(roll_out, pitch_out, yaw_velocity_out, z_velocity_out)

  #****************************************************************************
 
  # Publish Commands to the drone if we are not in hover mode.
  def SendCommand(self, roll, pitch, yaw_rate, z_dot):
    if not(math.isnan(pitch)):
      self.command.twist.linear.x = pitch
    else:
      print "**********************pitch is NaN****************************\n"
      self.command.twist.linear.x = 0

    if not(math.isnan(roll)):
      self.command.twist.linear.y = roll
    else:
      print "**********************roll is NaN**********************\n"
      self.command.twist.linear.y = 0

    if not(math.isnan(z_dot)):
      self.command.twist.linear.z = z_dot
    else:
      print "**********************z_dot is NaN**********************\n"
      self.command.twist.linear.z = 0

    if not(math.isnan(yaw_rate)):
      self.command.twist.angular.z = yaw_rate
    else:
      print "**********************yaw_rate is NaN**********************\n"
      self.command.twist.angular.z = 0

    # make sure the drone is not taking off
    if (self.status.drone_state != DroneStatus.TakingOff): 
      self.pubCommand.publish(self.command.twist)

    


  #***************    Callback Functions for Measurements   *******************

  def updateNavdata(self,nav_msg):
    self.status.drone_state = nav_msg.state

  #****************************************************************************
  # This method updates the current state of the drone
  def updateCurrentState(self,curr_data):

    # update the state information
    self.current_state.x      = curr_data.pos
    self.current_state.x_dot  = curr_data.vel
    self.current_state.x_ddot = curr_data.acc
    self.current_state.rpy    = curr_data.euler

    # Determine the commands to be sent to the drone
    self.determineCommands()

  #****************************************************************************

  # Update the desired state 
  def updateDesiredState(self,desiredState):

    # Update the desired state information
    self.desired_state.x      = np.array([desiredState.x,  desiredState.y,  desiredState.z],  dtype=np.float64)
    self.desired_state.x_dot  = np.array([desiredState.vx, desiredState.vy, desiredState.vz], dtype=np.float64)
    self.desired_state.x_ddot = np.array([desiredState.ax, desiredState.ay, desiredState.az], dtype=np.float64)
    self.desired_state.rpy    = np.array([desiredState.roll, desiredState.pitch, desiredState.yaw], dtype=np.float64)

  #****************************************************************************

  def clamp(self, num, upper=1.0, lower=None):
    if (lower is None):
      num = max(min(num,upper),-1.0*upper)
    else:
      num = max(min(num,upper),lower)
    return (num)

  #****************************************************************************

  # Sends a land signal to all flying drones
  def SendLandAll(self):
    self.pubLandAll.publish(Empty())
  
  # Sends a takeoff signal to all flying drones
  def SendTakeoffAll(self):
    self.pubTakeoffAll.publish(Empty())

  # Sends a savetofile signal to save d and u variables
  def SendSaveToFile(self):
    self.pubSaveToFile.publish(Empty())

  # Sends a readfromfile signal to read file with d and u variables
  def SendReadFromFile(self):
    self.pubReadFromFile.publish(Empty())

  # Send an emergency (or reset) message to the ardrone driver
  def SendEmergency(self):
    self.pubReset.publish(Empty())

  # Send a takeoff message to the ardrone driver
  def SendTakeoff(self,called=None):
    self.pubTakeoff.publish(Empty())
  
  # Send a landing message to the ardrone driver
  def SendLand(self,called=None):
    self.pubLand.publish(Empty())


  # Send Empty messages on /GoInitPos and /StartExp topics
  def sendGoInitPos(self):
    self.pubGoInitPos.publish(Empty())
    print "GoingToInitPos!"

  def sendStartExp(self):
    self.azhat = 0
    self.azhatdot = 0
    self.bzhat = 0 
    self.bzhatdot = 0
    self.pubStartExp.publish(Empty())
    #print "StartingExperiment!"

  #****************************************************************************
  # Keyboard Controller
  #****************************************************************************

  # This method is called when a key is pressed. It overrides the automated commands.
  def keyPressEvent(self, event):
    key = event.key()

    # If the key is not generated from an auto-repeating key
    if (not event.isAutoRepeat()):
      # Turn on override
      self.status.keyboard_override = 1 # turn on override
      roll_out = 0.0
      pitch_out = 0.0
      yaw_velocity_out = 0.0
      z_velocity_out = 0.0
      print "Key received"
      # Handle the important cases first!
      if key == KeyMapping.LandAll:
        self.SendLandAll()
      elif key == KeyMapping.TakeoffAll:
        self.SendTakeoffAll()
      elif key == KeyMapping.Emergency:
        self.SendEmergency()
      elif key == KeyMapping.Takeoff:
        self.SendTakeoff()
      elif key == KeyMapping.Land:
        self.SendLand()
      elif key == KeyMapping.StartHover:
        self.hover = 1
      elif key == KeyMapping.EndHover:
        self.hover = 0
      elif key == KeyMapping.SaveToFile:
        self.SendSaveToFile()
      elif key == KeyMapping.ReadFromFile:
        self.SendReadFromFile()
      elif key == KeyMapping.GoInitPos: # send empty messages on /function1 topic 
        self.sendGoInitPos() 
      elif key == KeyMapping.StartExp: # send empty messages on /function2 topic 
        print "Starting experiment :)-" 
        self.sendStartExp()
      else:
        # Now we handle moving, notice that this section is the opposite (+=) of the keyrelease section
        if key == KeyMapping.YawLeft:
          yaw_velocity_out = self.max_yaw
        elif key == KeyMapping.YawRight:
          yaw_velocity_out = -self.max_yaw
        elif key == KeyMapping.ForwardLeft:
          pitch_out = self.max_euler
          roll_out = self.max_euler
        elif key == KeyMapping.Forward:
          pitch_out = self.max_euler
        elif key == KeyMapping.ForwardRight:
          pitch_out = self.max_euler
          roll_out = -self.max_euler
        elif key == KeyMapping.Right:
          roll_out = -self.max_euler
        elif key == KeyMapping.BackwardRight:
          pitch_out = -self.max_euler
          roll_out = -self.max_euler
        elif key == KeyMapping.Backward:
          pitch_out = -self.max_euler
        elif key == KeyMapping.BackwardLeft:
          pitch_out = -self.max_euler
          roll_out = self.max_euler
        elif key == KeyMapping.Left:
          roll_out = self.max_euler
        elif key == KeyMapping.IncreaseAltitude:
          z_velocity_out = self.max_vz
        elif key == KeyMapping.DecreaseAltitude:
          z_velocity_out = -self.max_vz
      
      self.SendCommand(roll_out, pitch_out, yaw_velocity_out, z_velocity_out)

  #****************************************************************************

  def keyReleaseEvent(self,event):
    key = event.key()

    # If the key is not generated from an auto-repeating key
    if (not event.isAutoRepeat()):
      # Note that we don't handle the release of emergency/takeoff/landing keys here, there is no need.
      self.status.keyboard_override = 0 # turn off override
      self.SendCommand(0, 0, 0, 0)

#####################     Main Code to Run      ################################
if __name__=='__main__':

  # set up the signal handeler 
  signal.signal(signal.SIGINT, sigint_handler)

  # First we setup a ros node, so that we can communicate with the other packages
  rospy.init_node('nonlinear_controller')
  
  # Now we construct our Qt Application and associated windows
  app = QtGui.QApplication(sys.argv)
  display = DroneController()
  display.show()
  # executes the QT application
  status = app.exec_()

  # and only progresses to here once the application has been shutdown
  rospy.signal_shutdown('Great Flying!')
  sys.exit(status)
