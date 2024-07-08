import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import numba
from scipy.integrate import odeint

from scipy import integrate


Ca0 = 10
Cb0 = 10
length = 0.1  # [m]
ColD = 0.0212  # [m]
tinj = 100
# um = 0.00021263 # [m/s]
A = math.pi*ColD*ColD/4  # Column cross sectional area (m2)
um = 30.0/1000.0/60.0/1000/A  # [m/s]
eb = 0.62  # [-]
NX = 1000  # [-]
dx = length / NX
threshold_FL = 1.0e-8


@numba.jit("f8(f8[:],i8,f8)", nopython=True)
def flux_limiter_calculator(state, z_index, threshold):
    if (state[z_index + 1] - state[z_index]) < threshold and -threshold < (state[z_index + 1] - state[z_index]):
        if (state[z_index] - state[z_index - 1]) < threshold and -threshold < (state[z_index] - state[z_index - 1]): r = 0
        else: r = (state[z_index] - state[z_index - 1]) / (threshold)
    else: r = (state[z_index] - state[z_index - 1]) / (state[z_index + 1] - state[z_index])  # fluxlimitter

    return max(0, min(r, 1))  # minmod condition


@numba.jit("f8(f8,f8[:],i8,f8)", nopython=True)
def convective_flux(v, state, z_index, threshold):
    flux = v * (state[z_index] + 0.5 * flux_limiter_calculator(state, z_index, threshold) * (
                state[z_index + 1] - state[z_index]))
    return flux


@numba.jit("f8(f8,f8,f8,f8)", nopython=True)
def flux_limiter_calculator_specific_state(state_back, state_center, state_forward, threshold):
    if (state_forward - state_center) < threshold and -threshold < (state_forward - state_center):
        if (state_center - state_back) < threshold and -threshold < (state_center - state_back): r = 0
        else: r = (state_center - state_back) / (threshold)
    else: r = (state_center - state_back) / (state_forward - state_center)  # fluxlimitter

    return max(0, min(r, 1))  # minmod condition


@numba.jit("f8(f8,f8,f8,f8,f8)", nopython=True)
def convective_flux_half_forward_specific_state(v, state_back, state_center, state_forward, threshold):
    flux_half_forward = v * (
                state_center + 0.5 * flux_limiter_calculator_specific_state(state_back, state_center, state_forward,
                                                                            threshold) * (state_forward - state_center))
    return flux_half_forward


@numba.jit("f8[:](f8,f8[:],f8,f8,f8,f8,f8,f8,f8,f8,f8,f8)", nopython=True)
def sim_model_flux_limiter(t, x, H1, H2, K1, K2, b1, b2, dx, eb, um, threshold):

    v = um / eb
    alpha = 1 / dx
    beta = (1 - eb) / eb

    c1 = x[0:NX]
    q1 = x[NX:NX + NX]
    c2 = x[2 * NX:3 * NX]
    q2 = x[3 * NX:4 * NX]
    dxdt = np.empty(4 * NX)

    dc1dt = dxdt[0:NX]
    dc2dt = dxdt[2 * NX:3 * NX]
    dq1dt = dxdt[NX:NX + NX]
    dq2dt = dxdt[3 * NX:4 * NX]

    c1_in = Ca0 * (t <= tinj)
    c2_in = Cb0 * (t <= tinj)

    for i in range(0, NX):
        qeq1 = H1 * c1[i] / (1.0 + b1 * c1[i] + b2 * c2[i])
        qeq2 = H2 * c2[i] / (1.0 + b1 * c1[i] + b2 * c2[i])
        dq1dt[i] = K1 * (qeq1 - q1[i])
        dq2dt[i] = K2 * (qeq2 - q2[i])

    dc1dt[0] = alpha * (
                v * c1_in - convective_flux_half_forward_specific_state(v, c1_in, c1[0], c1[1], threshold)) - beta * \
               dq1dt[0]
    dc2dt[0] = alpha * (
                v * c2_in - convective_flux_half_forward_specific_state(v, c2_in, c2[0], c2[1], threshold)) - beta * \
               dq2dt[0]
    dc1dt[NX - 1] = alpha * (convective_flux(v, c1, NX - 2, threshold) - v * c1[NX - 1]) - beta * dq1dt[NX - 1]
    dc2dt[NX - 1] = alpha * (convective_flux(v, c2, NX - 2, threshold) - v * c2[NX - 1]) - beta * dq2dt[NX - 1]

    for i in range(1, NX - 1):
        dc1dt[i] = alpha * (convective_flux(v, c1, i - 1, threshold) - convective_flux(v, c1, i, threshold)) - beta * \
                   dq1dt[i]
        dc2dt[i] = alpha * (convective_flux(v, c2, i - 1, threshold) - convective_flux(v, c2, i, threshold)) - beta * \
                   dq2dt[i]

    return dxdt


def my_model_deconvoluted(theta, t):
    nt = len(t)
    output = np.zeros((nt, 2))
    x0 = np.zeros(4 * NX)
    output[0, :] = 0
    H1, H2, K1, K2, b1, b2 = theta

    sol = integrate.odeint(sim_model_flux_limiter, x0, t, args=(H1, H2, K1, K2, b1, b2, dx, eb, um, threshold_FL), tfirst=True)
    # sol1 = integrate.solve_ivp(sim_model,[0,t[-1]],x0,method="LSODA",t_eval=t,args=(H1,H2,Kapp1,Kapp2,dx,eb,um))
    # sol = sol1.y.T

    return sol


H1_true, H2_true = [5.97, 8.52]  # Henry's Constant [A, B]
k1_true, k2_true = [0.81, 0.569]  # Overall Mass Transfer Coefficient [A, B]
b1_true, b2_true = [0.154, 0.295]  # Affinity Coefficient [A, B]

theta_t = np.array([H1_true, H2_true, k1_true, k2_true, b1_true, b2_true])
t_list = np.linspace(0, 300, 1001)

# truemodel = my_model(theta_t, t_list)
data = my_model_deconvoluted(theta_t, t_list)
Ca = []
Cb = []
x = []
t = []


for i in range(1000):
    Ca = np.hstack((Ca, data[:, i]))
    Cb = np.hstack((Cb, data[:, 2000+i]))
    x = np.hstack((x, np.ones(1001)*(length/NX)*i*100))
    t = np.hstack((t, np.linspace(0, 10, 1001)))

FEM = np.vstack((x, t, Ca, Cb)).T
np.savetxt("FEM_forward_Ha=5.97 Hb=8.52.dat", FEM, delimiter="\t", fmt="%f")
