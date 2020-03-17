# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:32:49 2020

@author: Sandro
"""

import numpy as np    
from random import gauss
import pdb


# Frame conversion (3,2,1 Euler Angles)
def C_1(ang):
    ca = np.cos(ang)
    sa = np.sin(ang)
    C = np.array([[1,   0,  0],
                  [0,  ca, sa],
                  [0, -sa, ca]])
    return C
def C_2(ang):
    ca = np.cos(ang)
    sa = np.sin(ang)
    C = np.array([[ca, 0, -sa],
                  [0,  1,   0],
                  [sa, 0,  ca]])
    return C
def C_3(ang):
    ca = np.cos(ang)
    sa = np.sin(ang)
    C = np.array([[ca,  sa, 0],
                  [-sa, ca, 0],
                  [0,    0, 1]])
    return C
def C_321(ang):
    a1 = ang[0]
    a2 = ang[1]
    a3 = ang[2]
    C = C_1(a1)*C_2(a2)*C_3(a3)
    return C

# IMU propogation ODE model
def ode(x,IMUacc_mes,IMUw_mes):
    # Extract states (ang;pos;vel)
    IMUang_NED = x[0:3]
    IMUvel_relNED_NED = x[6:9]
    # Extract measurements
    IMUw_relNED_IMU = IMUw_mes
    IMUacc_relNED_IMU = IMUacc_mes
    # Rotational kinematics
    S_inv = np.array([[1, np.sin(IMUang_NED[0])*np.tan(IMUang_NED[1]), np.cos(IMUang_NED[0])*np.tan(IMUang_NED[1])],
                      [0, np.cos(IMUang_NED[0]),                      -np.sin(IMUang_NED[0])],
                      [0, np.sin(IMUang_NED[0])/np.cos(IMUang_NED[1]), np.cos(IMUang_NED[0])/np.cos(IMUang_NED[1])]])
    IMUang_NED_dot = np.dot(S_inv,IMUw_relNED_IMU)
    # Translational kinematics
    IMUpos_relNED_NED_dot = IMUvel_relNED_NED
    C_IMU_NED = C_321(IMUang_NED)
    C_NED_IMU = np.transpose(C_IMU_NED)
    IMUvel_relNED_NED_dot = np.dot(C_NED_IMU,IMUacc_relNED_IMU)
    IMUvel_relNED_NED_dot[2] = IMUvel_relNED_NED_dot[2] - 9.81
    
    #pdb.set_trace()
    x_dot = np.concatenate((IMUang_NED_dot,IMUpos_relNED_NED_dot,IMUvel_relNED_NED_dot),axis=0)
    return x_dot

# Forward Euler integrator (ode1)
def propogate(f,t_ini,t_step,t_fin,x_ini,par_1,par_2):
    x_len = x_ini.size
    t_len = int((t_fin-t_ini)/t_step+1)
    x_out = np.zeros((x_len, t_len))
    x = x_ini
    x_out[0:x_len+1,0] = x
    for i in range(0,t_len-1):
        IMUacc_NED_mes = par_1[0:3,i]
        IMUw_NED_mes = par_2[0:3,i]
        x_dot = f(x,IMUacc_NED_mes,IMUw_NED_mes)
        x = x + t_step*x_dot
        x_out[0:x_len+1,i+1] = x
    return x_out

# %% Generate states and measurements %% #
def gen_meas_circle(t_ini, t_step, t_fin, r, SNR):
    # Time
    t_len = int((t_fin-t_ini)/t_step+1)
    time = np.linspace(t_ini,t_fin,t_len) # (s)
    tau = t_fin/2/np.pi

    # State space model
    acc_mean = 0
    acc_std = 1/np.sqrt(SNR)*r/(tau**2)/2
    grav_acc = 9.81
    IMUpos_NED = np.zeros((3, t_len))
    IMUvel_NED = np.zeros((3, t_len))
    IMUacc_NED = np.zeros((3, t_len))
    IMUacc_mes = np.zeros((3, t_len))
    IMUang_NED = np.zeros((3, t_len))
    IMUw_mes = np.zeros((3, t_len))
    for i in range(0, t_len):
        # True states
        IMUpos_NED[0,i] = r*np.cos(time[i]/tau)
        IMUpos_NED[1,i] = r*np.sin(time[i]/tau)
        IMUvel_NED[0,i] = -r/tau*np.sin(time[i]/tau)
        IMUvel_NED[1,i] = r/tau*np.cos(time[i]/tau)
        IMUacc_NED[0,i] = -r/(tau**2)*np.cos(time[i]/tau)
        IMUacc_NED[1,i] = -r/(tau**2)*np.sin(time[i]/tau)    
        # Measurements
        C_IMU_NED = C_321(np.zeros((3,1)))
        IMUacc_mes[0:3,i] = np.dot(C_IMU_NED,IMUacc_NED[0:3,i])
        IMUacc_mes[0,i] = IMUacc_mes[0,i] + gauss(acc_mean,acc_std)
        IMUacc_mes[1,i] = IMUacc_mes[1,i] + gauss(acc_mean,acc_std)
        IMUacc_mes[2,i] = IMUacc_mes[2,i] + gauss(acc_mean,acc_std) + grav_acc
        IMUw_mes[0,i] = gauss(acc_mean,acc_std)/10
        IMUw_mes[1,i] = gauss(acc_mean,acc_std)/10
        IMUw_mes[2,i] = gauss(acc_mean,acc_std)/10

    x_ini = np.concatenate((IMUang_NED[0:3,0],IMUpos_NED[0:3,0],IMUvel_NED[0:3,0]),axis=0)
    return time, IMUacc_mes, IMUw_mes, x_ini

def read_meas_data(data):
    freq = 200
    t_len = len(data)
    t_ini = 0
    t_fin = (t_len-1)/freq
    time = np.linspace(t_ini,t_fin,t_len)
    acc_meas = np.zeros((3, t_len))
    #acc_ngrav_meas = np.zeros((3,t_len))
    gyr_meas = np.zeros((3, t_len))
    #qua_meas = np.zeros((4,t_len))
    #mag_meas = np.zeros((3,t_len))
    #eul_meas = np.zeros((3,t_len))
    temp = np.zeros((3,1))
    for i in range(0,t_len-1):
        #acc_true_meas = data[i][0]
        acc_meas[0][i] = data[i][1][0]
        acc_meas[1][i] = data[i][1][1]
        acc_meas[2][i] = data[i][1][2]
        gyr_meas[0][i] = data[i][2][0]
        gyr_meas[1][i] = data[i][2][1]
        gyr_meas[2][i] = data[i][2][2]
        #qua_meas = data[i][3]
        #mag_meas = data[i][4]
        #eul_meas = data[i][5]
    x_ini = np.concatenate((temp[0:3,0],temp[0:3,0],temp[0:3,0]),axis=0)
    return time, acc_meas, gyr_meas, x_ini

