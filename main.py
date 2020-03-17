# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:07:47 2020

@author: Sandro
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json

import imulib as imu


# %% Generate IMU measurements %% #

# Select measurements to use for propogation
mode = 1
class measurement_type:
    import_data = 1
    circle_model = 2
    swing_model = 3 
    
# Define parameters and initialize measurement data
if mode == measurement_type.import_data:
    print('IMU Measurement Imported')
    #str_key = "circle01"
    str_key = "static01"
    data_path = 'sensor_data'+os.sep+str_key+'.json'
    with open(data_path,"r") as read_file:
        data = json.load(read_file)
    time, IMUacc_relNED_IMU, IMUw_relNED_IMU, x_ini = imu.read_meas_data(data)
    t_ini = time[0]
    t_step = time[1]
    t_len = time.size
    t_fin = time[t_len-1]
        
elif mode == measurement_type.circle_model:
    print('Circle Trajectory Model')
    str_key = "circle_model"
    t_ini = 0
    t_step = 0.01
    t_fin = 2
    radius = 1
    SNR = 6
    time, IMUacc_relNED_IMU, IMUw_relNED_IMU, x_ini = imu.gen_meas_circle(t_ini, t_step, t_fin, radius, SNR)
    t_len = time.size
    
elif mode == measurement_type.swing_model:
    print('Golf Swing Trajectory Model')
    str_key = "swing_model"
    
else:
    print('Warning: Undefined Mode')


# %% Propogate state from measurements %% #

# Initialize 
x = np.zeros((9, t_len))
x[0:9,0] = x_ini

# Propogate
x[0:9,0:t_len] = imu.propogate(imu.ode,t_ini,t_step,t_fin,x_ini,IMUacc_relNED_IMU,IMUw_relNED_IMU)

# %% Post-processing data %% #

# Extract states
IMUang_NED = x[0:3,0:t_len]
IMUpos_relNED_NED = x[3:6,0:t_len]
IMUvel_relNED_NED = x[6:9,0:t_len]

# Rotate measurements to inertial frame
IMUacc_relNED_NED = np.zeros((3, t_len))
IMUw_relNED_NED = np.zeros((3, t_len))
for i in range(0,t_len-1):
    C_IMU_NED = imu.C_321(IMUang_NED[0:3,i])
    C_NED_IMU = np.transpose(C_IMU_NED)
    IMUacc_relNED_NED[0:3,i] = np.dot(C_NED_IMU,IMUacc_relNED_IMU[0:3,i])
    IMUw_relNED_NED[0:3,i] = np.dot(C_NED_IMU,IMUw_relNED_IMU[0:3,i])
    IMUacc_relNED_NED[2,i] = IMUacc_relNED_NED[2,i] - 9.81


# %% Plot results %%  

# Set up plot folder
pwd_path = os.getcwd()
plot_path = pwd_path+os.sep+'figs_'+str_key+os.sep
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

# IMU acceleration measurement vs time
fig, ax = plt.subplots()
ax.plot(time[0:t_len-1],IMUacc_relNED_NED[0,0:t_len-1], label='NEDx')
ax.plot(time[0:t_len-1],IMUacc_relNED_NED[1,0:t_len-1], label='NEDy')
ax.plot(time[0:t_len-1],IMUacc_relNED_NED[2,0:t_len-1], label='NEDz')
ax.set_xlabel('time (s)')
ax.set_ylabel('acc (m/s2)')
ax.set_title("IMU Accelerometer Measurements")
ax.legend()
plt.savefig(plot_path+'meas_acc_time.pdf', bbox_inches='tight')

# IMU acceleration measurement vs time
fig, ax = plt.subplots()
ax.plot(time,IMUw_relNED_NED[0,0:t_len+1], label='NEDx')
ax.plot(time,IMUw_relNED_NED[1,0:t_len+1], label='NEDy')
ax.plot(time,IMUw_relNED_NED[2,0:t_len+1], label='NEDz')
ax.set_xlabel('time (s)')
ax.set_ylabel('w (rad/s)')
ax.set_title("IMU Gyroscope Measurements")
ax.legend()
plt.savefig(plot_path+'meas_w_time.pdf', bbox_inches='tight')

# Propogated trajectory (3D)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(IMUpos_relNED_NED[0,0:t_len+1], IMUpos_relNED_NED[1,0:t_len+1], IMUpos_relNED_NED[2,0:t_len+1])
ax.set_xlabel('NEDx (m)')
ax.set_ylabel('NEDy (m)')
ax.set_zlabel('NEDz (m)');
ax.set_title("Propogated Trajectory")
plt.savefig(plot_path+'state_traj.pdf')

# Propogated trajectory (2D projection)
fig, ax = plt.subplots()
ax.plot(IMUpos_relNED_NED[0,0:t_len+1],IMUpos_relNED_NED[1,0:t_len+1])
ax.set_xlabel('NEDx (m)')
ax.set_ylabel('NEDy (m)')
ax.set_title("Propogated Trajectory (2D projection)")
plt.savefig(plot_path+'state_traj_2dproj.pdf', bbox_inches='tight')

# Propogated 3-2-1 euler angles vs time
fig, ax = plt.subplots()
ax.plot(time,IMUang_NED[0,0:t_len+1], label='roll')
ax.plot(time,IMUang_NED[1,0:t_len+1], label='pitch')
ax.plot(time,IMUang_NED[2,0:t_len+1], label='yaw')
ax.set_xlabel('time (s)')
ax.set_ylabel('Euler Angle (rad)')
ax.set_title("Propogated Orientation")
ax.legend()
plt.savefig(plot_path+'state_ang_time.pdf', bbox_inches='tight')

# Propogated position vs time
fig, ax = plt.subplots()
ax.plot(time,IMUpos_relNED_NED[0,0:t_len+1], label='NEDx')
ax.plot(time,IMUpos_relNED_NED[1,0:t_len+1], label='NEDy')
ax.plot(time,IMUpos_relNED_NED[2,0:t_len+1], label='NEDz')
ax.set_xlabel('time (s)')
ax.set_ylabel('dist (m)')
ax.set_title("Propogated Position")
ax.legend()
plt.savefig(plot_path+'state_pos_time.pdf', bbox_inches='tight')

# Propogated velocity vs time
fig, ax = plt.subplots()
ax.plot(time,IMUvel_relNED_NED[0,0:t_len+1], label='NEDx')
ax.plot(time,IMUvel_relNED_NED[1,0:t_len+1], label='NEDy')
ax.plot(time,IMUvel_relNED_NED[2,0:t_len+1], label='NEDz')
ax.set_xlabel('time (s)')
ax.set_ylabel('dist (m)')
ax.set_title("Propogated Velocity")
ax.legend()
plt.savefig(plot_path+'state_vel_time.pdf', bbox_inches='tight')

