# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:07:47 2020

@author: Sandro
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import imulib as imu


# %% Generate IMU measurements %% #

# Select measurement type
mode = 2
class measurement_type:
    import_data = 1
    circle_model = 2
    swing_model = 3 
    
# Define parameters and initialize measurement data
if mode == measurement_type.import_data:
    print('IMU Measurement Imported')
    str1 = "fname"
    
elif mode == measurement_type.circle_model:
    print('Circle Trajectory Model')
    str1 = "circle_model"
    t_ini = 0
    t_step = 0.01
    t_fin = 2
    radius = 1
    SNR = 6
    time, IMUacc_NED_mes, IMUw_NED_mes, x_ini = imu.gen_meas_circle(t_ini, t_step, t_fin, radius, SNR)
    
elif mode == measurement_type.swing_model:
    print('Golf Swing Trajectory Model')
    str1 = "swing_model"
    
else:
    print('Warning: Undefined Mode')


# %% Propogate state from measurements %% #

# Initialize 
t_len = time.size
x = np.zeros((9, t_len))
x[0:9,0] = x_ini

# Propogate
x[0:9,0:t_len] = imu.propogate(imu.ode,t_ini,t_step,t_fin,x_ini,IMUacc_NED_mes,IMUw_NED_mes)

# %% Post-processing Data %% #

pwd_path = os.getcwd()
plot_path = pwd_path+os.sep+str1+'_figs'+os.sep
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

# %% Plot results %%  

# IMU acceleration measurement vs time
fig, ax = plt.subplots()
ax.plot(time,IMUacc_NED_mes[0,0:t_len+1], label='NEDx')
ax.plot(time,IMUacc_NED_mes[1,0:t_len+1], label='NEDy')
ax.plot(time,IMUacc_NED_mes[2,0:t_len+1], label='NEDz')
ax.set_xlabel('time (s)')
ax.set_ylabel('acc (m/s^2)')
ax.set_title("IMU Acceleration Measurements")
ax.legend()
plt.savefig(plot_path+'meas_acc_time.pdf', bbox_inches='tight')

# IMU acceleration measurement vs time
fig, ax = plt.subplots()
ax.plot(time,IMUw_NED_mes[0,0:t_len+1], label='NEDx')
ax.plot(time,IMUw_NED_mes[1,0:t_len+1], label='NEDy')
ax.plot(time,IMUw_NED_mes[2,0:t_len+1], label='NEDz')
ax.set_xlabel('time (s)')
ax.set_ylabel('acc (m/s^2)')
ax.set_title("IMU Acceleration Measurements")
ax.legend()
plt.savefig(plot_path+'meas_w_time.pdf', bbox_inches='tight')

# Propogated trajectory (3D)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x[3,0:t_len+1], x[4,0:t_len+1], x[5,0:t_len+1])
ax.set_xlabel('NEDx (m)')
ax.set_ylabel('NEDy (m)')
ax.set_zlabel('NEDz (m)');
ax.set_title("Propogated Trajectory")
plt.savefig(plot_path+'state_traj.pdf', bbox_inches='tight')

# Propogated position vs time
fig, ax = plt.subplots()
ax.plot(x[3,0:t_len+1],x[4,0:t_len+1])
ax.set_xlabel('NEDx (m)')
ax.set_ylabel('NEDy (m)')
ax.set_title("Propogated Trajectory (2D projection)")
plt.savefig(plot_path+'state_traj_2dproj.pdf', bbox_inches='tight')

# Propogated position vs time
fig, ax = plt.subplots()
ax.plot(time,x[0,0:t_len+1], label='roll')
ax.plot(time,x[1,0:t_len+1], label='pitch')
ax.plot(time,x[2,0:t_len+1], label='yaw')
ax.set_xlabel('time (s)')
ax.set_ylabel('Euler Angle (deg)')
ax.set_title("Propogated Orientation")
ax.legend()
plt.savefig(plot_path+'state_ang_time.pdf', bbox_inches='tight')

# Propogated position vs time
fig, ax = plt.subplots()
ax.plot(time,x[3,0:t_len+1], label='NEDx')
ax.plot(time,x[4,0:t_len+1], label='NEDy')
ax.plot(time,x[5,0:t_len+1], label='NEDz')
ax.set_xlabel('time (s)')
ax.set_ylabel('dist (m)')
ax.set_title("Propogated Position")
ax.legend()
plt.savefig(plot_path+'state_pos_time.pdf', bbox_inches='tight')

# Propogated position vs time
fig, ax = plt.subplots()
ax.plot(time,x[6,0:t_len+1], label='NEDx')
ax.plot(time,x[7,0:t_len+1], label='NEDy')
ax.plot(time,x[8,0:t_len+1], label='NEDz')
ax.set_xlabel('time (s)')
ax.set_ylabel('dist (m)')
ax.set_title("Propogated Velocity")
ax.legend()
plt.savefig(plot_path+'state_vel_time.pdf', bbox_inches='tight')

