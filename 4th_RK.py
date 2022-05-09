
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 12:25:09 2020

@author: grknk
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pandas as pd
import statistics
import os


h = 0.01
n = int(400 / h)  # steps per episode (4000)
t = np.zeros(n + 1)  # Initiate time
# theta=np.zeros(n+1)                         #Initiate theta
# state_theta = np.zeros(n+1)                 #Initiate state theta
x = np.zeros(n + 1)  # Initiate x
z = np.zeros(n + 1)  # Initiate z
px = np.zeros(n + 1)  # Initiate px
pz = np.zeros(n + 1)  # Initiate pz
D_R = 0.01  # Rotational diffusivity for calculating noise for theta
D_0 = 0.001  # Translational diffusivity for calculating noise for x and z

nv_x = np.zeros(n + 1)
nv_z = np.zeros(n + 1)
nv_px = np.zeros(n + 1)
nv_pz = np.zeros(n + 1)

t[0] = 0  # Initial cond for t
# state_theta[0] = math.pi/4                  #Initial cond for state theta

# x[0]= -math.pi/2                            #Initial cond for x
# z[0]= -math.pi/2                            #Initial cond for z
x[0] = 6 * math.pi  # Initial cond for x
z[0] = 0  # Initial cond for z
px[0] = 1 / math.sqrt(2)  # Initial cond for px
pz[0] = 1 / math.sqrt(2)  # Initial cond for pz

nv_x[0] = 3*np.pi
nv_z[0] = 0
nv_px[0] = 1 / math.sqrt(2)
nv_pz[0] = 1 / math.sqrt(2)

phi = 0.3
psi = 1

state_size = 12  # 12 possible states for worticity and state theta
action_size = 4  # Four possible actions
q_table = np.zeros((state_size, action_size))  # Initaite the table for learning
q_table += (n*2*math.pi)

# q_table += n*4*math.pi
# print(q_table)
# q_table += 10**8

learning_rate = 0.14  # ♫alpha value
exploration_rate = 0.001  # epsilon value
discount = 0.999  # gamma value


def state_value(vorticity, state_theta):  # Used to find which of 12 states the particle is in
    state = int()

    if (vorticity <= 1 and vorticity > 0.33) and (state_theta > 7 * math.pi / 4 and state_theta < 9 * math.pi / 4):
        state = 0

    elif (vorticity <= 1 and vorticity > 0.33) and (state_theta > -math.pi / 4 and state_theta < math.pi / 4):
        state = 0

    elif (vorticity <= 0.33 and vorticity >= -0.33) and (
            state_theta > 7 * math.pi / 4 and state_theta < 9 * math.pi / 4):
        state = 1

    elif (vorticity <= 0.33 and vorticity >= -0.33) and (state_theta > -math.pi / 4 and state_theta < math.pi / 4):
        state = 1

    elif (vorticity < -0.33 and vorticity >= -1) and (state_theta > 7 * math.pi / 4 and state_theta < 9 * math.pi / 4):
        state = 2

    elif (vorticity < -0.33 and vorticity >= -1) and (state_theta > -math.pi / 4 and state_theta < math.pi / 4):
        state = 2

    elif (vorticity <= 1 and vorticity > 0.33) and (state_theta > 5 * math.pi / 4 and state_theta < 7 * math.pi / 4):
        state = 3

    elif (vorticity <= 1 and vorticity > 0.33) and (state_theta > -3 * math.pi / 4 and state_theta < -math.pi / 4):
        state = 3

    elif (vorticity <= 0.33 and vorticity >= -0.33) and (
            state_theta > 5 * math.pi / 4 and state_theta < 7 * math.pi / 4):
        state = 4

    elif (vorticity <= 0.33 and vorticity >= -0.33) and (state_theta > -3 * math.pi / 4 and state_theta < -math.pi / 4):
        state = 4

    elif (vorticity < -0.33 and vorticity >= -1) and (state_theta > 5 * math.pi / 4 and state_theta < 7 * math.pi / 4):
        state = 5

    elif (vorticity < -0.33 and vorticity >= -1) and (state_theta > -3 * math.pi / 4 and state_theta < -math.pi / 4):
        state = 5

    elif (vorticity <= 1 and vorticity > 0.33) and (state_theta > 3 * math.pi / 4 and state_theta < 5 * math.pi / 4):
        state = 6

    elif (vorticity <= 1 and vorticity > 0.33) and (state_theta > -5 * math.pi / 4 and state_theta < -3 * math.pi / 4):
        state = 6

    elif (vorticity <= 0.33 and vorticity >= -0.33) and (
            state_theta > 3 * math.pi / 4 and state_theta < 5 * math.pi / 4):
        state = 7

    elif (vorticity <= 0.33 and vorticity >= -0.33) and (
            state_theta > -5 * math.pi / 4 and state_theta < -3 * math.pi / 4):
        state = 7

    elif (vorticity < -0.33 and vorticity >= -1) and (state_theta > 3 * math.pi / 4 and state_theta < 5 * math.pi / 4):
        state = 8

    elif (vorticity < -0.33 and vorticity >= -1) and (
            state_theta > -5 * math.pi / 4 and state_theta < -3 * math.pi / 4):
        state = 8

    elif (vorticity <= 1 and vorticity > 0.33) and [
        (state_theta >= 1 * math.pi / 4 and state_theta < 3 * math.pi / 4) or (
                state_theta > -7 * math.pi / 4 and state_theta < -5 * math.pi / 4)]:
        state = 9

    elif (vorticity <= 0.33 and vorticity >= -0.33) and [
        (state_theta >= 1 * math.pi / 4 and state_theta < 3 * math.pi / 4) or (
                state_theta > -7 * math.pi / 4 and state_theta < -5 * math.pi / 4)]:
        state = 10

    elif (vorticity < -0.33 and vorticity >= -1) and [
        (state_theta >= 1 * math.pi / 4 and state_theta < 3 * math.pi / 4) or (
                state_theta > -7 * math.pi / 4 and state_theta < -5 * math.pi / 4)]:
        state = 11

    return state


def probabilities(newstate):  # Probability of taking the new action at the new state, eq 3 in gparticle paper
    optimal_action = np.argmax(q_table[newstate, :])  # index of the best action, 0 1 2 3
    p0 = exploration_rate / 4
    p1 = exploration_rate / 4
    p2 = exploration_rate / 4
    p3 = exploration_rate / 4

    if optimal_action == 0:
        p0 += 1 - exploration_rate

    elif optimal_action == 1:
        p1 += 1 - exploration_rate

    elif optimal_action == 2:
        p2 += 1 - exploration_rate

    elif optimal_action == 3:
        p3 += 1 - exploration_rate

    return (p0, p1, p2, p3)


def action(action_theta):  # Used to find which action the particle takes

    action = int()
    k_xnew = int()
    k_znew = int()

    if action_theta >= 1 * math.pi / 4 and action_theta < 3 * math.pi / 4:
        action = 0
        k_xnew = 0
        k_znew = 1

    elif action_theta > 7 * math.pi / 4 and action_theta < 9 * math.pi / 4:
        action = 1
        k_xnew = 1
        k_znew = 0

    elif action_theta > 5 * math.pi / 4 and action_theta < 7 * math.pi / 4:
        action = 2
        k_xnew = 0
        k_znew = -1

    elif action_theta > 3 * math.pi / 4 and action_theta < 5 * math.pi / 4:
        action = 3
        k_xnew = -1
        k_znew = 0

    return (action, k_xnew, k_znew)


reward_current_episode = 0
naive_reward_this_episode = 0

noisetheta = np.random.normal(0,1,n)
noisepx = np.random.normal(0, 1, n)
noisepz = np.random.normal(0, 1, n)
noisex = np.random.normal(0, 1, n)
noisez = np.random.normal(0, 1, n)

nv_noisepx = np.random.normal(0, 1, n)
nv_noisepz = np.random.normal(0, 1, n)
nv_noisex = np.random.normal(0, 1, n)
nv_noisez = np.random.normal(0, 1, n)

px_noise = np.zeros(n + 1)
pz_noise = np.zeros(n + 1)
x_noise = np.zeros(n + 1)
z_noise = np.zeros(n + 1)

nv_px_noise = np.zeros(n + 1)
nv_pz_noise = np.zeros(n + 1)
nv_x_noise = np.zeros(n + 1)
nv_z_noise = np.zeros(n + 1)


episode_count = 1
rewards_all_episodes = np.zeros(episode_count)

k_ax = 0
k_az = 1
nv_k_ax = 0
nv_k_az = 1

# noise = np.random.normal(0,1,n)        #değiştir
theta = np.zeros(n + 1)
theta[0] = math.atan2(pz[0], px[0])

ensemble = np.zeros(4501)
ensemble2 = np.zeros(4501)


for j in range(episode_count):

    x = np.zeros(n + 1)  # Initiate x
    z = np.zeros(n + 1)  # Initiate z
    px = np.zeros(n + 1)  # Initiate px
    pz = np.zeros(n + 1)  # Initiate pz
    t = np.zeros(n + 1)

    # nv_x = np.zeros(n + 1)
    # nv_z = np.zeros(n + 1)
    # nv_px = np.zeros(n + 1)
    # nv_pz = np.zeros(n + 1)

    x[0] = np.random.uniform(0, 2 * math.pi)
    z[0] = np.random.uniform(0, 2 * math.pi)
    # x[0] = np.random.uniform(2*math.pi,4*math.pi)
    # z[0] = np.random.uniform(2*math.pi,4*math.pi)
    px[0] = np.random.uniform(0, 1)
    pz[0] = math.sqrt(1 - px[0] ** 2)
    t[0] = 0
    theta[0] = math.atan2(pz[0], px[0])
    k_ax = 0
    k_az = 1

    # nv_x[0] = x[0]
    # nv_z[0] = z[0]
    # nv_px[0] = px[0]
    # nv_pz[0] = pz[0]
    action_matrix = np.zeros(n + 1)
    action_matrix[0] = int(0)

    reward_current_episode = 0
    naive_reward_this_episode = 0

    for i in range(0, n):


        px_noise[i] = noisepx[i] * (2 * D_R * h) ** 0.5
        pz_noise[i] = noisepz[i] * (2 * D_R * h) ** 0.5
        x_noise[i] = noisex[i] * (2 * D_0 * h) ** 0.5
        z_noise[i] = noisez[i] * (2 * D_0 * h) ** 0.5

        nv_px_noise[i] = nv_noisepx[i] * (2 * D_R * h) ** 0.5
        nv_pz_noise[i] = nv_noisepz[i] * (2 * D_R *h)**0.5
        nv_x_noise[i] = nv_noisex[i] * (2 * D_0 * h) ** 0.5
        nv_z_noise[i] = nv_noisez[i] * (2 * D_0 * h) ** 0.5

        # nv_px_noise[i] = px_noise[i]
        # nv_pz_noise[i] = pz_noise[i]
        # nv_x_noise[i] = x_noise[i]
        # nv_z_noise[i] = z_noise[i]



        fx = phi * px[i] - (1 / 2) * np.cos(x[i]) * np.sin(z[i])
        fz = phi * pz[i] + (1 / 2) * np.sin(x[i]) * np.cos(z[i])
        f_px = 1 / (2 * psi) * (k_ax - (k_ax * px[i] + k_az * pz[i]) * px[i]) - np.cos(x[i]) * np.cos(z[i]) * pz[i] / 2
        f_pz = 1 / (2 * psi) * (k_az - (k_ax * px[i] + k_az * pz[i]) * pz[i]) + (
                np.cos(x[i]) * np.cos(z[i]) * px[i]) / 2

        k_1px = f_px
        k_1pz = f_pz
        k_1x = fx
        k_1z = fz

        k_2px = 1 / (2 * psi) * (k_ax - (k_ax * (px[i] + k_1px * 0.5 * h + px_noise[i] * 0.5) + k_az * (pz[i] + k_1pz * 0.5 * h + pz_noise[i] * 0.5)) * (
                px[i] + k_1px * 0.5 * h + px_noise[i] * 0.5 )) - (np.cos(x[i] + k_1x * 0.5 * h + x_noise[i] * 0.5) * np.cos(z[i] + k_1z * 0.5 * h + z_noise[i] * 0.5) * (
                pz[i] + k_1pz * 0.5 * h + pz_noise[i]*0.5)) / 2
        k_2pz = 1 / (2 * psi) * (k_az - (k_ax * (px[i] + k_1px * 0.5 * h + px_noise[i] * 0.5) + k_az * (pz[i] + k_1pz * 0.5 * h + pz_noise[i] * 0.5)) * (
                pz[i] + k_1pz * 0.5 * h + pz_noise[i] * 0.5)) + (np.cos(x[i] + k_1x * 0.5 * h + x_noise[i] * 0.5) * np.cos(z[i] + k_1z * 0.5 * h + z_noise[i] * 0.5) * (
                px[i] + k_1px * 0.5 * h + px_noise[i] * 0.5)) / 2
        k_2x = -(1 / 2) * np.cos(x[i] + k_1x * 0.5 * h + x_noise[i] * 0.5) * np.sin(z[i] + k_1z * 0.5 * h + z_noise[i] * 0.5) + phi * (
                    px[i] + k_1px * 0.5 * h + px_noise[i] * 0.5)
        k_2z = (1 / 2) * np.sin(x[i] + k_1x * 0.5 * h + x_noise[i] * 0.5) * np.cos(z[i] + k_1z * 0.5 * h + z_noise[i] * 0.5) + phi * (pz[i] + k_1pz * 0.5 * h + pz_noise[i] * 0.5)


        k_3px = 1 / (2 * psi) * (k_ax - (k_ax * (px[i] + k_2px * 0.5 * h + px_noise[i] * 0.5) + k_az * (
                    pz[i] + k_2pz * 0.5 * h + pz_noise[i] * 0.5)) * (
                                         px[i] + k_2px * 0.5 * h + px_noise[i] * 0.5)) - (
                            np.cos(x[i] + k_2x * 0.5 * h + x_noise[i] * 0.5) * np.cos(
                        z[i] + k_2z * 0.5 * h + z_noise[i] * 0.5) * (
                                    pz[i] + k_2pz * 0.5 * h + pz_noise[i] * 0.5)) / 2
        k_3pz = 1 / (2 * psi) * (k_az - (k_ax * (px[i] + k_2px * 0.5 * h + px_noise[i] * 0.5) + k_az * (
                    pz[i] + k_2pz * 0.5 * h + pz_noise[i] * 0.5)) * (
                                         pz[i] + k_2pz * 0.5 * h + pz_noise[i] * 0.5)) + (
                            np.cos(x[i] + k_2x * 0.5 * h + x_noise[i] * 0.5) * np.cos(
                        z[i] + k_2z * 0.5 * h + z_noise[i] * 0.5) * (
                                    px[i] + k_2px * 0.5 * h + px_noise[i] * 0.5)) / 2
        k_3x = -(1 / 2) * np.cos(x[i] + k_2x * 0.5 * h + x_noise[i] * 0.5) * np.sin(
            z[i] + k_2z * 0.5 * h + z_noise[i] * 0.5) + phi * (
                       px[i] + k_2px * 0.5 * h + px_noise[i] * 0.5)
        k_3z = (1 / 2) * np.sin(x[i] + k_2x * 0.5 * h + x_noise[i] * 0.5) * np.cos(
            z[i] + k_2z * 0.5 * h + z_noise[i] * 0.5) + phi * (pz[i] + k_2pz * 0.5 * h + pz_noise[i] * 0.5)


        k_4px = 1 / (2 * psi) * (k_ax - (k_ax * (px[i] + k_3px * h + px_noise[i]) + k_az * (pz[i] + k_3pz * h + pz_noise[i])) * (
                px[i] + k_3px * h + px_noise[i])) - (np.cos(x[i] + k_3x * h + x_noise[i]) * np.cos(z[i] + k_3z * h + z_noise[i]) * (
                pz[i] + k_3pz * h + pz_noise[i])) / 2
        k_4pz = 1 / (2 * psi) * (k_az - (k_ax * (px[i] + k_3px * h + px_noise[i]) + k_az * (pz[i] + k_3pz * h + pz_noise[i])) * (
                pz[i] + k_3pz * h + pz_noise[i])) + (np.cos(x[i] + k_3x * h + x_noise[i]) * np.cos(z[i] + k_3z * h + z_noise[i]) * (
                px[i] + k_3px * h + px_noise[i])) / 2
        k_4x = -(1 / 2) * np.cos(x[i] + k_3x * h + x_noise[i]) * np.sin(z[i] + k_3z * h + z_noise[i]) + phi * (
                    px[i] + k_3px * h + px_noise[i])
        k_4z = (1 / 2) * np.sin(x[i] + k_3x * h + x_noise[i]) * np.cos(z[i] + k_3z * h + z_noise[i]) + phi * (pz[i] + k_3pz * h + pz_noise[i])

        # k_3px = 1 / (2 * psi) * (k_ax - (k_ax * (px[i] + 0.5 * k_2px * h) + k_az * (pz[i] + 0.5 * k_2pz * h)) * (
        #         px[i] + 0.5 * k_2px * h)) - (np.cos(x[i] + 0.5 * k_2x * h) * np.cos(z[i] + 0.5 * k_2z * h) * (
        #         pz[i] + 0.5 * k_2pz * h)) / 2
        # k_3pz = 1 / (2 * psi) * (k_az - (k_ax * (px[i] + 0.5 * k_2px * h) + k_az * (pz[i] + 0.5 * k_2pz * h)) * (
        #         pz[i] + 0.5 * k_2pz * h)) + (np.cos(x[i] + 0.5 * k_2x * h) * np.cos(z[i] + 0.5 * k_2z * h) * (
        #         px[i] + 0.5 * k_2px * h)) / 2
        # k_3x = -(1 / 2) * np.cos(x[i] + 0.5 * k_2x * h) * np.sin(z[i] + 0.5 * k_2z * h) + phi * (
        #             px[i] + 0.5 * k_2px * h)
        # k_3z = (1 / 2) * np.sin(x[i] + 0.5 * k_2x * h) * np.cos(z[i] + 0.5 * k_2z * h) + phi * (pz[i] + 0.5 * k_2pz * h)
        #
        # k_4px = 1 / (2 * psi) * (
        #         k_ax - (k_ax * (px[i] + k_3px * h) + k_az * (pz[i] + k_3pz * h)) * (px[i] + k_3px * h)) - (
        #                 np.cos(x[i] + k_3x * h) * np.cos(z[i] + k_3z * h) * (pz[i] + k_3pz * h)) / 2
        # k_4pz = 1 / (2 * psi) * (
        #         k_az - (k_ax * (px[i] + k_3px * h) + k_az * (pz[i] + k_3pz * h)) * (pz[i] + k_3pz * h)) + (
        #                 np.cos(x[i] + k_3x * h) * np.cos(z[i] + k_3z * h) * (px[i] + k_3px * h)) / 2
        # k_4x = -(1 / 2) * np.cos(x[i] + k_3x * h) * np.sin(z[i] + k_3z * h) + phi * (px[i] + k_3px * h)
        # k_4z = (1 / 2) * np.sin(x[i] + k_3x * h) * np.cos(z[i] + k_3z * h) + phi * (pz[i] + k_3pz * h)

        px[i + 1] = px[i] + (k_1px + k_2px + k_3px + k_4px) * h / 6 + px_noise[i]
        pz[i + 1] = pz[i] + (k_1pz + k_2pz + k_3pz + k_4pz) * h / 6 + pz_noise[i]
        x[i + 1] = x[i] + (k_1x + k_2x + k_3x + k_4x) * h / 6 + x_noise[i]
        z[i + 1] = z[i] + (k_1z + k_2z + k_3z + k_4z) * h / 6 + z_noise[i]
        t[i + 1] = t[i] + h

        px[i + 1] = px[i + 1] / (math.sqrt(px[i + 1] ** 2 + pz[i + 1] ** 2))
        pz[i + 1] = pz[i + 1] / (math.sqrt(px[i + 1] ** 2 + pz[i + 1] ** 2))


        theta[i + 1] = math.atan2(pz[i + 1], px[i + 1])

        nv_eqn1 = (1 / (2 * psi)) * (nv_k_ax - (nv_k_ax * nv_px[i] + nv_k_az * nv_pz[i]) * nv_px[i]) - (
                np.cos(nv_x[i]) * np.cos(nv_z[i]) * nv_pz[i]) / 2  # px
        nv_eqn2 = (1 / (2 * psi)) * (nv_k_az - (nv_k_ax * nv_px[i] + nv_k_az * nv_pz[i]) * nv_pz[i]) + (
                np.cos(nv_x[i]) * np.cos(nv_z[i]) * nv_px[i]) / 2  # pz
        nv_eqn3 = -(1 / 2) * np.cos(nv_x[i]) * np.sin(nv_z[i]) + phi * nv_px[i]  # fx
        nv_eqn4 = (1 / 2) * np.sin(nv_x[i]) * np.cos(nv_z[i]) + phi * nv_pz[i]  # fz

        nv_k_1px = nv_eqn1
        nv_k_1pz = nv_eqn2
        nv_k_1x = nv_eqn3
        nv_k_1z = nv_eqn4

        # nv_k_2px = (1 / (2 * psi)) * (nv_k_ax - (
        #         nv_k_ax * (nv_px[i] + nv_k_1px * h + nv_px_noise[i]) + nv_k_az * (nv_pz[i] + nv_k_1pz * h + nv_pz_noise[i])) * (
        #                                       nv_px[i] + nv_k_1px * h + nv_px_noise[i])) - (
        #                    np.cos(nv_x[i] + nv_k_1x * h + nv_x_noise[i]) * np.cos(nv_z[i] + nv_k_1z * h + nv_z_noise[i]) * (
        #                    nv_pz[i] + nv_k_1pz * h + nv_pz_noise[i])) / 2
        # nv_k_2pz = (1 / (2 * psi)) * (nv_k_az - (
        #         nv_k_ax * (nv_px[i] + nv_k_1px * h + nv_px_noise[i]) + nv_k_az * (nv_pz[i] + nv_k_1pz * h + nv_pz_noise[i])) * (
        #                                       nv_pz[i] + nv_k_1pz * h + nv_pz_noise[i])) + (
        #                    np.cos(nv_x[i] + nv_k_1x * h + nv_x_noise[i]) * np.cos(nv_z[i] + nv_k_1z * h + nv_z_noise[i]) * (
        #                    nv_px[i] + nv_k_1px * h + nv_px_noise[i])) / 2
        # nv_k_2x = -(1 / 2) * np.cos(nv_x[i] + nv_k_1x * h + nv_x_noise[i]) * np.sin(nv_z[i] + nv_k_1z * h + nv_z_noise[i]) + phi * (
        #         nv_px[i] + nv_k_1px * h + nv_px_noise[i])
        # nv_k_2z = (1 / 2) * np.sin(nv_x[i] + nv_k_1x * h + nv_x_noise[i]) * np.cos(nv_z[i] + nv_k_1z * h + nv_z_noise[i]) + phi * (
        #         nv_pz[i] + nv_k_1pz * h + nv_pz_noise[i])

        nv_k_2px = (1 / (2 * psi)) * (nv_k_ax - (
                nv_k_ax * (nv_px[i] + 0.5 * nv_k_1px * h + nv_px_noise[i] * 0.5) + nv_k_az * (nv_pz[i] + 0.5 * nv_k_1pz * h + nv_pz_noise[i] * 0.5)) * (
                                              nv_px[i] + 0.5 * nv_k_1px * h + nv_px_noise[i] * 0.5)) - (
                           np.cos(nv_x[i] + 0.5 * nv_k_1x * h + nv_x_noise[i] * 0.5) * np.cos(nv_z[i] + 0.5 * nv_k_1z * h + nv_z_noise[i] * 0.5) * (
                           nv_pz[i] + 0.5 * nv_k_1pz * h + nv_pz_noise[i] * 0.5)) / 2
        nv_k_2pz = (1 / (2 * psi)) * (nv_k_az - (
                nv_k_ax * (nv_px[i] + 0.5 * nv_k_1px * h + nv_px_noise[i] * 0.5) + nv_k_az * (nv_pz[i] + 0.5 * nv_k_1pz * h + nv_pz_noise[i] * 0.5)) * (
                                              nv_pz[i] + 0.5 * nv_k_1pz * h + nv_pz_noise[i] * 0.5)) + (
                           np.cos(nv_x[i] + 0.5 * nv_k_1x * h + nv_x_noise[i] * 0.5) * np.cos(nv_z[i] + 0.5 * nv_k_1z * h + nv_z_noise[i] * 0.5) * (
                           nv_px[i] + 0.5 * nv_k_1px * h + nv_px_noise[i] * 0.5)) / 2
        nv_k_2x = -(1 / 2) * np.cos(nv_x[i] + 0.5 * nv_k_1x * h + nv_x_noise[i] * 0.5) * np.sin(nv_z[i] + 0.5 * nv_k_1z * h + nv_z_noise[i] * 0.5) + phi * (
                nv_px[i] + 0.5 * nv_k_1px * h + nv_px_noise[i] * 0.5)
        nv_k_2z = (1 / 2) * np.sin(nv_x[i] + 0.5 * nv_k_1x * h + nv_x_noise[i] * 0.5) * np.cos(nv_z[i] + 0.5 * nv_k_1z * h + nv_z_noise[i] * 0.5) + phi * (
                nv_pz[i] + 0.5 * nv_k_1pz * h + nv_pz_noise[i] * 0.5)

        nv_k_3px = (1 / (2 * psi)) * (nv_k_ax - (
                nv_k_ax * (nv_px[i] + 0.5 * nv_k_2px * h + nv_px_noise[i] * 0.5) + nv_k_az * (nv_pz[i] + 0.5 * nv_k_2pz * h + nv_pz_noise[i] * 0.5)) * (
                                              nv_px[i] + 0.5 * nv_k_2px * h + nv_px_noise[i] * 0.5)) - (
                           np.cos(nv_x[i] + 0.5 * nv_k_2x * h + nv_x_noise[i] * 0.5) * np.cos(nv_z[i] + 0.5 * nv_k_2z * h + nv_z_noise[i] * 0.5) * (
                           nv_pz[i] + 0.5 * nv_k_2pz * h + nv_pz_noise[i] * 0.5)) / 2
        nv_k_3pz = (1 / (2 * psi)) * (nv_k_az - (
                nv_k_ax * (nv_px[i] + 0.5 * nv_k_2px * h + nv_px_noise[i] * 0.5) + nv_k_az * (nv_pz[i] + 0.5 * nv_k_2pz * h + nv_pz_noise[i] * 0.5)) * (
                                              nv_pz[i] + 0.5 * nv_k_2pz * h + nv_pz_noise[i] * 0.5)) + (
                           np.cos(nv_x[i] + 0.5 * nv_k_2x * h + nv_x_noise[i] * 0.5) * np.cos(nv_z[i] + 0.5 * nv_k_2z * h + nv_z_noise[i] * 0.5) * (
                           nv_px[i] + 0.5 * nv_k_2px * h + nv_px_noise[i] * 0.5)) / 2
        nv_k_3x = -(1 / 2) * np.cos(nv_x[i] + 0.5 * nv_k_2x * h + nv_x_noise[i] * 0.5) * np.sin(nv_z[i] + 0.5 * nv_k_2z * h + nv_z_noise[i] * 0.5) + phi * (
                nv_px[i] + 0.5 * nv_k_2px * h + nv_px_noise[i] * 0.5)
        nv_k_3z = (1 / 2) * np.sin(nv_x[i] + 0.5 * nv_k_2x * h + nv_x_noise[i] * 0.5) * np.cos(nv_z[i] + 0.5 * nv_k_2z * h + nv_z_noise[i] * 0.5) + phi * (
                nv_pz[i] + 0.5 * nv_k_2pz * h + nv_pz_noise[i] * 0.5)

        nv_k_4px = (1 / (2 * psi)) * (
                nv_k_ax - (nv_k_ax * (nv_px[i] + nv_k_3px * h + nv_px_noise[i]) + nv_k_az * (nv_pz[i] + nv_k_3pz * h + nv_pz_noise[i])) * (
                nv_px[i] + nv_k_3px * h + nv_px_noise[i])) - (
                           np.cos(nv_x[i] + nv_k_3x * h + nv_x_noise[i]) * np.cos(nv_z[i] + nv_k_3z * h + nv_z_noise[i]) * (
                           nv_pz[i] + nv_k_3pz * h + nv_pz_noise[i])) / 2
        nv_k_4pz = (1 / (2 * psi)) * (
                nv_k_az - (nv_k_ax * (nv_px[i] + nv_k_3px * h + nv_px_noise[i]) + nv_k_az * (nv_pz[i] + nv_k_3pz * h + nv_pz_noise[i])) * (
                nv_pz[i] + nv_k_3pz * h + nv_pz_noise[i])) + (
                           np.cos(nv_x[i] + nv_k_3x * h + nv_x_noise[i]) * np.cos(nv_z[i] + nv_k_3z * h + nv_z_noise[i]) * (
                           nv_px[i] + nv_k_3px * h + nv_px_noise[i])) / 2
        nv_k_4x = -(1 / 2) * np.cos(nv_x[i] + nv_k_3x * h + nv_x_noise[i]) * np.sin(nv_z[i] + nv_k_3z * h + nv_z_noise[i]) + phi * (
                nv_px[i] + nv_k_3px * h + nv_px_noise[i])
        nv_k_4z = (1 / 2) * np.sin(nv_x[i] + nv_k_3x * h + nv_x_noise[i]) * np.cos(nv_z[i] + nv_k_3z * h + nv_z_noise[i]) + phi * (
                nv_pz[i] + nv_k_3pz * h + nv_pz_noise[i])

        nv_px[i + 1] = nv_px[i] + (nv_k_1px + nv_k_2px + nv_k_3px + nv_k_4px) * h / 6 + nv_px_noise[i]
        nv_pz[i + 1] = nv_pz[i] + (nv_k_1pz + nv_k_2pz + nv_k_3pz + nv_k_4pz) * h / 6 + nv_pz_noise[i]
        nv_x[i + 1] = nv_x[i] + (nv_k_1x + nv_k_2x + nv_k_3x + nv_k_4x) * h / 6 + nv_x_noise[i]
        nv_z[i + 1] = nv_z[i] + (nv_k_1z + nv_k_2z + nv_k_3z + nv_k_4z) * h / 6 + nv_z_noise[i]

        nv_px[i + 1] = nv_px[i + 1] / (math.sqrt(nv_px[i + 1] ** 2 + nv_pz[i + 1] ** 2))
        nv_pz[i + 1] = nv_pz[i + 1] / (math.sqrt(nv_px[i + 1] ** 2 + nv_pz[i + 1] ** 2))

        vort_start = -np.cos(x[i]) * np.cos(z[i])  # vorticity at the current state
        dir_start = theta[i]  # direction (state theta) at current state
        # dir_start = arctan_vel

        vort_new = -np.cos(x[i + 1]) * np.cos(z[i + 1])  # vorticity at new state
        dir_new = theta[i + 1]  # direction at new state
        action_start = int(action_matrix[i])  # action taken to get to the new state (an)

        state_start = state_value(vort_start, dir_start)  # first state
        state_new = state_value(vort_new, dir_new)  # new state
        reward = z[i + 1] - z[i]
        naive_reward = nv_z[i + 1] - nv_z[i]

        q_table[state_start, action_start] = q_table[state_start, action_start] * (
                    1 - learning_rate) + learning_rate * (reward + discount * np.max(q_table[state_new, :]))

        p0, p1, p2, p3 = probabilities(state_new)
        new_action = np.random.choice(
            [random.uniform(math.pi / 4, 3 * math.pi / 4), random.uniform(7 * math.pi / 4, 9 * math.pi / 4),
             random.uniform(5 * math.pi / 4, 7 * math.pi / 4), random.uniform(3 * math.pi / 4, 5 * math.pi / 4)],
            p=[p0, p1, p2, p3])
        action_matrix[i + 1], k_ax, k_az = action(new_action)

        reward_current_episode += reward
        naive_reward_this_episode += naive_reward
        # print(state_start, action_start, state_new)

    # rewards_all_episodes[j] = abs((reward_current_episode -naive_reward_this_episode) / naive_reward_this_episode)
    rewards_all_episodes[j] = reward_current_episode

    # x = np.mod(x + np.pi,2*np.pi) - np.pi
    # z = np.mod(z + np.pi, 2*np.pi) - np.pi

    # if j in (0, 1, 2, 3, 4, 49, 99, 149, 199, 249, 299, 349, 399, 449, 499, 549, 599,
    #          649, 699, 749, 799, 849, 899, 949, 999):
    #     plt.figure(j+1)
    #     plt.clf()
    #     plt.scatter(x, z, s=0.6, c="b")
    #     plt.title(label=('$\phi$ =' + str(phi) + ' $\psi$ = ' + str(psi) + ' $\epsilon$ = ' + str(
    #         exploration_rate) + ' Case C' + '\n' + 'Episode: ' + str(j+1)))
    #     plt.xlabel('x')
    #     plt.ylabel('z')
    #     #
    #     axes = plt.gca()
    #     # axes.set_xlim([-6 * math.pi, 3 * math.pi])
    #     # axes.set_xlim([0, 24 * math.pi])
    #     # axes.set_ylim([-math.pi, 20 * math.pi])
    #     # axes.set_ylim([-6 * math.pi, 18 * math.pi])

    #     # plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
    #     # plt.xticks([0, np.pi * 6, np.pi * 12], [r'$0$', r'$6\pi$', r'$0$'])
    #     # plt.xticks([0, np.pi * 6, np.pi * 12, np.pi * 18, np.pi * 24],
    #     #             [r'$0$', r'$6\pi$', r'$12\pi$', r'$18\pi$', r'$24\pi$'])
    #     # plt.yticks([-np.pi * 6, 0, np.pi * 6, np.pi * 12, np.pi * 18],
    #     #             [r'$-6\pi$', r'$0$', r'$6\pi/2$', r'$12\pi$', r'$18\pi$'])

    #     # plt.xlim(0, 24 * np.pi)
    #     # plt.ylim(-np.pi * 6, np.pi * 18)
    #     plt.gca().set_aspect('equal', adjustable='box')

    #     plt.grid(linestyle='--')

    #     plt.legend(bbox_to_anchor=(0, -0.2, 1, 0), loc=2, ncol=1, mode="expand", borderaxespad=0)

    if j >= 499:
        ensemble[j-499] = sum(rewards_all_episodes[j-499:j+1])/4500
        
        


    # print(q_table)


panda_table = pd.DataFrame(q_table, columns=['Up', 'Right', 'Down', "Left"],
                           index=['w+ right', 'w0 right', 'w- right', "w+ down", "w0 down", "w- down", "w+ left",
                                  "w0 left", "w- left", "w+ up", "w0 up", "w- up"])
panda_table.to_csv(r'C:/Users/grknk/reinforceddata.csv', header=True, index=True)


# print(panda_table)
# print("reward current episode= ",reward_current_episode)

a1 = []
b1 = []
listepisode = []
for i in range(0,len(x)-1):
    a1.append(x[i+1] - x[i])
    b1.append(z[i+1] - z[i])
    listepisode.append(i+1) 
    
plt.figure()
plt.plot(a1, c='r', label = 'in x direction')
plt.title('Difference from previous position')


plt.figure()
plt.plot(b1, c='b', label = 'in z direction')
plt.title('Difference from previous position')



x = np.mod(x + np.pi,2*np.pi) - np.pi
z = np.mod(z + np.pi, 2*np.pi) - np.pi

plt.figure(5000)
plt.clf()
plt.scatter(x, z, s=0.6, c='b')
#plt.plot(nv_x, nv_z, 'r-.')
plt.title(label=('$\phi$ =' + str(phi) + ' $\psi$ = ' + str(psi) + ' $\epsilon$ = ' + str(exploration_rate) + ' Case C' + '\n' + 'Episode: ' + str(episode_count)))
plt.xlabel('x') 
plt.ylabel('z')
#
axes = plt.gca()
plt.gca().set_aspect('equal', adjustable='box')


plt.grid(linestyle='--')

plt.legend(bbox_to_anchor=(0, -0.2, 1, 0), loc=2, ncol=1, mode="expand", borderaxespad=0)
plt.show()

plt.figure(2)
plt.clf()
plt.plot(rewards_all_episodes)
plt.show()

plt.figure(3) 
plt.clf()
plt.plot(np.arange(500,5001), ensemble)

 

plt.show()




# plt.figure(3)
# plt.clf()
# plt.plot(x_store,z_store,'b:')
# plt.plot(nv_x_store,nv_z_store,'r--')
# plt.title(label = ('$\phi$ =' + str(phi) + ' $\psi$ = ' + str(psi) + ' $\epsilon$ = ' + str(exploration_rate) + ' Case C' + '\n' + 'Episode: ' + str(episode_count)))
# plt.xlabel('x')
# plt.ylabel('z')
#
# axes = plt.gca()
# axes.set_xlim([-6*math.pi, 3*math.pi])
# axes.set_xlim([0, 24*math.pi])
# axes.set_ylim([-math.pi,20*math.pi])
# axes.set_ylim([-6*math.pi,18*math.pi])

# plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
# plt.xticks([0, np.pi*6, np.pi*12],[r'$0$', r'$6\pi$',  r'$0$'])
# plt.xticks([0, np.pi*6, np.pi*12, np.pi*18, np.pi*24],[r'$0$', r'$6\pi$',  r'$12\pi$', r'$18\pi$', r'$24\pi$'])
# plt.yticks([-np.pi*6, 0, np.pi*6, np.pi*12, np.pi*18],[r'$-6\pi$', r'$0$', r'$6\pi/2$', r'$12\pi$', r'$18\pi$'])

# plt.xlim(0,12*np.pi)
# plt.ylim(-np.pi*6,np.pi*18)
# plt.gca().set_aspect('equal', adjustable='box')
#
#
# plt.grid(linestyle='--')
#
# plt.legend(bbox_to_anchor=(0, -0.2, 1, 0), loc=2, ncol=1, mode="expand", borderaxespad=0)
# plt.show()
"""
"""
# path="D:\\00000REAL\\hopper\\hopperInitialState"
# for i in range(1,401):
    
#     b = i*100
#     os.chdir(path)
#     Newfolder=str(i*0.25)
#     os.makedirs(Newfolder)
    
#     path2= path + "\\" +Newfolder
#     os.chdir(path2)
#     Newfolder_2 = "lagrangian"
#     os.makedirs(Newfolder_2)
     
    
#     path3= path2 + "\\" +Newfolder_2
#     os.chdir(path3)
#     Newfolder_3 = "kinematicCloud"
#     os.makedirs(Newfolder_3)
    
    
#     path4= path3 + "\\" +Newfolder_3
#     os.chdir(path4)
#     Newfolder_4 = "constant"
#     os.makedirs(Newfolder_4)
    
#     f= open("positions","w+")    
#     line = 'FoamFile {   version     2.0;    format      ascii;    class       Cloud<basicKinematicCollidingParcel>;     location    ' +'"'+ str(i)+'/lagrangian/kinematicCloud'+'";     object      positions;    }' +'1((' + str(x[b-1]) + ' ' + str(z[b-1]) + ' ' + '0.003' + ')' + ' 1)'
#     f.write(line + "\n")
#     f.close() 

# path="D:\\00000REAL\\hopper\\diminishing"
# f= open("kinematicCloudPositions","w+") #bunu kinematicCloudPositions'a


# line = "("
# f.write(line + "\n")
# for i in range(1,40001):
    
#     line = "(" + str(x[i]) + " " + str(z[i])+ " " + str(0) + ")"
#     f.write(line + "\n")
# line = ")"   
# f.write(line + "\n")
# f.close()

