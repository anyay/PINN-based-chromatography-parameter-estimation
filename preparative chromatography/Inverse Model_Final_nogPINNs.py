from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import sys
import re

import deepxde as dde
from deepxde.backend import tf
import math
import pandas as pd
import random

random.seed(12899)

L = 0.1  # Column length(m)
d = 0.0212  # Column diameter(m)
A = np.pi*d*d/4  # Column cross sectional area (m2)
e = 0.62  # porosity
v = 30.0/1000.0/1000/60.0/A/e  # Velocity (m/s)
f = 12  # Feed concentration (-)
te = 500  # final time (s)
t_inj1 = 20  # first injected time for component A and B (s)
t_inj2 = 200  # second injected time for component A and B (s)
t_inj3 = 250


# Scale the problem
scale_x = 100
scale_t = 1/50
scale_y = 1
L_scaled = L * scale_x
v_scaled = v * scale_x / scale_t
t_scaled = te * scale_t
t_inj1_scaled = t_inj1 * scale_t
t_inj2_scaled = t_inj2 * scale_t
t_inj3_scaled = t_inj3 * scale_t



# Datasets
data1 = pd.read_csv(r"C:\Users\10716\OneDrive\桌面\DeepXDE\DeepXDE\SMB\FEM_two injection_Noisy3rd_2000points_A.csv")
data2 = pd.read_csv(r"C:\Users\10716\OneDrive\桌面\DeepXDE\DeepXDE\SMB\FEM_two injection_Noisy3rd_2000points_B.csv")

np.save(r"C:\Users\10716\OneDrive\桌面\DeepXDE\DeepXDE\SMB\NoisyDataset_two injection_3rd_2000points_A", data1)
np.save(r"C:\Users\10716\OneDrive\桌面\DeepXDE\DeepXDE\SMB\NoisyDataset_two injection_3rd_2000points_B", data2)

traindata1 = np.load(r"C:\Users\10716\OneDrive\桌面\DeepXDE\DeepXDE\SMB\NoisyDataset_two injection_3rd_2000points_A.npy")
traindata2 = np.load(r"C:\Users\10716\OneDrive\桌面\DeepXDE\DeepXDE\SMB\NoisyDataset_two injection_3rd_2000points_B.npy")


xx1 = traindata1[1:, 2:3]
tt1 = traindata1[1:, 0:1]
Ca = traindata1[1:, 1:2]
xx2 = traindata2[1:, 2:3]
tt2 = traindata2[1:, 0:1]
Cb = traindata2[1:, 1:2]

observe_x1, observe_Ca = np.hstack((scale_x * xx1, scale_t * tt1)), Ca
observe_x2, observe_Cb = np.hstack((scale_x * xx2, scale_t * tt2)), Cb

observe_y1 = dde.icbc.PointSetBC(observe_x1, Ca, component=0)
observe_y2 = dde.icbc.PointSetBC(observe_x2, Cb, component=1)

observe_x = np.vstack((observe_x1, observe_x2))


# Non-uniform distribution for the left boundary condition
z1 = np.linspace(0, (t_inj1 / 500) * L_scaled, num=200)
z2 = np.linspace((t_inj2 / 500) * L_scaled, (t_inj3 / 500) * L_scaled, num=500)
z = np.hstack((z1, z2))
t = 0
Z, T = np.meshgrid(z, t)
Training_Points = np.vstack((Z.flatten(), T.flatten())).T

Extra_Points = np.vstack((observe_x, Training_Points))


# Constraint the search range of unknown parameters
ka_scaled_ = dde.Variable(random.uniform(0, 1) / scale_t)  # Mass transfer coefficient of A (1/s)
kb_scaled_ = dde.Variable(random.uniform(0, 1) / scale_t)  # Mass transfer coefficient of B (1/s)

Ha_ = dde.Variable(random.uniform(0, 10))  # Henry constant of A (-)
Hb_ = dde.Variable(random.uniform(0, 10))  # Henry constant of B (-)

ba_ = dde.Variable(random.uniform(0, 1))  # equilibrium coefficient
bb_ = dde.Variable(random.uniform(0, 1))  # equilibrium coefficient
"""
ka_scaled = (0.319 / scale_t) * tf.tanh(ka_scaled_) + (0.319 / scale_t)
kb_scaled = (0.19 / scale_t) * tf.tanh(kb_scaled_) + (0.19 / scale_t)

Ha = 3.23 * tf.tanh(Ha_) + 3.23
Hb = 6.5 * tf.tanh(Hb_) + 6.5

ba = 0.21 * tf.tanh(ba_) + 0.21
bb = 0.422 * tf.tanh(bb_) + 0.422
"""

geom = dde.geometry.Interval(0, L_scaled)
timedomain = dde.geometry.TimeDomain(0, t_scaled)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# PINNs
def pde(x, y):
    y_star = scale_y * y
    ca, cb, qa, qb = y_star[:, 0:1], y_star[:, 1:2], y_star[:, 2:3], y_star[:, 3:4]

    ca_x = dde.grad.jacobian(y_star, x, i=0, j=0)
    ca_t = dde.grad.jacobian(y_star, x, i=0, j=1)

    cb_x = dde.grad.jacobian(y_star, x, i=1, j=0)
    cb_t = dde.grad.jacobian(y_star, x, i=1, j=1)

    qa_t = dde.grad.jacobian(y_star, x, i=2, j=1)

    qb_t = dde.grad.jacobian(y_star, x, i=3, j=1)

    massbalance_liquid_a = (
            ca_t + (1 - e) / e * qa_t + v_scaled * ca_x
    )
    massbalance_liquid_b = (
            cb_t + (1 - e) / e * qb_t + v_scaled * cb_x
    )
    massbalance_solid_a = (
            ka_scaled_ * ((Ha_ * ca) / (1 + ba_ * ca + bb_ * cb) - qa) - qa_t
    )
    massbalance_solid_b = (
            kb_scaled_ * ((Hb_ * cb) / (1 + ba_ * ca + bb_ * cb) - qb) - qb_t
    )
    return [massbalance_liquid_a, massbalance_liquid_b, massbalance_solid_a, massbalance_solid_b]


def feed_concentration(x):
    # Define the conditions for the piecewise function
    conditions = [
        x[:, 1:] <= t_inj1_scaled,
        (x[:, 1:] > t_inj1_scaled) & (x[:, 1:] <= t_inj2_scaled),
        (x[:, 1:] > t_inj2_scaled) & (x[:, 1:] <= t_inj3_scaled),
        x[:, 1:] > t_inj3_scaled
    ]
    # x, t = np.split(x, 2, axis=1)
    return np.piecewise(x[:, 1:], conditions, [lambda a: f, lambda a: 0, lambda a: f, lambda a: 0])


def boundary_beg(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


bc_beg_ca = dde.DirichletBC(
    geomtime, feed_concentration, boundary_beg, component=0
)
bc_beg_cb = dde.DirichletBC(
    geomtime, feed_concentration, boundary_beg, component=1
)

initial_condition_ca = dde.IC(
    geomtime, lambda x: 0, lambda _, on_initial: on_initial, component=0
)
initial_condition_cb = dde.IC(
    geomtime, lambda x: 0, lambda _, on_initial: on_initial, component=1
)
initial_condition_qa = dde.IC(
    geomtime, lambda x: 0, lambda _, on_initial: on_initial, component=2
)
initial_condition_qb = dde.IC(
    geomtime, lambda x: 0, lambda _, on_initial: on_initial, component=3
)


data = dde.data.TimePDE(
    geomtime,
    pde,
    [initial_condition_ca,
     initial_condition_cb,
     initial_condition_qa,
     initial_condition_qb,
     bc_beg_ca,
     bc_beg_cb,
     observe_y1,
     observe_y2],
    num_domain=1500,
    num_initial=300,
    num_boundary=1200,
    anchors=Extra_Points,
)
layer_size = [2] + [27] * 4 + [4]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
# net.apply_output_transform(lambda x, y: y * scale_y)

gPINNmodel = dde.Model(data, net)
resampler = dde.callbacks.PDEResidualResampler(period=100)
gPINNmodel.compile("adam", lr=0.0001, external_trainable_variables=[ka_scaled_, kb_scaled_, Ha_, Hb_, ba_, bb_], loss_weights=[30, 10, 1e-2, 1e-2, 1, 1, 1, 1, 1, 1, 100, 100])
variable = dde.callbacks.VariableValue([ka_scaled_ * scale_t, kb_scaled_ * scale_t, Ha_, Hb_, ba_, bb_], period=1000, filename="variables.dat")
losshistory, train_state = gPINNmodel.train(epochs=300000, callbacks=[variable, resampler], disregard_previous_best=True)


# plots
"""Get the domain: x = L_scaled and t from 0 to t_scaled"""
X_nn = L_scaled * np.ones((100, 1))
T_nn = np.linspace(0, t_scaled, 100).reshape(100, 1)
X_pred = np.append(X_nn, T_nn, axis=1)

y_pred = gPINNmodel.predict(X_pred)
ca_pred, cb_pred, qa_pred, qb_pred = y_pred[:, 0:1], y_pred[:, 1:2], y_pred[:, 2:3], y_pred[:, 3:]
plt.figure()
plt.plot(T_nn / scale_t, ca_pred, color='blue', linewidth=3., label='Concentration A')
plt.plot(T_nn / scale_t, cb_pred, color='red', linewidth=3., label='Concentration B')
# plt.plot(X_pred, qa_pred)
# plt.plot(X_pred, qb_pred)

plt.legend()
plt.grid(True)
plt.xlabel('t')
plt.ylabel('Concentration')

plt.show()

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
