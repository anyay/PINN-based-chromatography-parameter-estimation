#%%

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize

import random
import numba
import time
import datetime

from scipy import integrate

Ca0 = 12
Cb0 = 12
length = 0.1  # [m]
ColD = 0.0212  # [m]
tinj1 = 20
tinj2 = 200
tinj3 = 250
# um = 0.00021263 # [m/s]
A = math.pi*ColD*ColD/4  # Column cross sectional area (m2)
um = 30.0/1000.0/60.0/1000/A  # [m/s]
eb = 0.62  # [-]
NX_3rd = 50  # [-]
NX_flux = 300  # [-]
dx_3rd = length / NX_3rd
dx_flux = length / NX_flux
threshold_FL = 1.0e-8

random.seed(12905)
ka_initial = random.uniform(0, 1)  # Mass transfer coefficient of A (1/s)
kb_initial = random.uniform(0, 1)  # Mass transfer coefficient of B (1/s)
Ha_initial = random.uniform(0, 10)  # Henry constant of A (-)
Hb_initial = random.uniform(0, 10)  # Henry constant of B (-)
ba_initial = random.uniform(0, 1)  # equilibrium coefficient
bb_initial = random.uniform(0, 1)  # equilibrium coefficient

@numba.jit("f8[:](f8,f8[:],f8[:],f8[:])", nopython=True)
def sim_model_3rdFEM(t, x, theta, params):
    H1, H2, K1, K2, b1, b2 = theta
    dx_3rd, eb, um, Ca0, Cb0, tinj1, tinj2, tinj3 = params

    alpha = um / (6.0 * dx_3rd * eb)
    beta = (1 - eb) / eb

    c1 = x[0:NX_3rd]
    q1 = x[NX_3rd:NX_3rd + NX_3rd]
    c2 = x[2 * NX_3rd:3 * NX_3rd]
    q2 = x[3 * NX_3rd:4 * NX_3rd]
    dxdt = np.empty(4 * NX_3rd)

    dc1dt = dxdt[0:NX_3rd]
    dc2dt = dxdt[2 * NX_3rd:3 * NX_3rd]
    dq1dt = dxdt[NX_3rd:NX_3rd + NX_3rd]
    dq2dt = dxdt[3 * NX_3rd:4 * NX_3rd]

    c1_in = Ca0 * (t <= tinj1 or (t >= tinj2 and t <= tinj3))
    c2_in = Cb0 * (t <= tinj1 or (t >= tinj2 and t <= tinj3))

    for i in range(0, NX_3rd):
        qeq1 = H1 * c1[i] / (1.0 + b1 * c1[i] + b2 * c2[i])
        qeq2 = H2 * c2[i] / (1.0 + b1 * c1[i] + b2 * c2[i])
        dq1dt[i] = K1 * (qeq1 - q1[i])
        dq2dt[i] = K2 * (qeq2 - q2[i])

        # dc1dt[0] = -alpha * (2 * c1[1] + 3 * c1[0] - 6 * c1_in + c1_in) - beta * dq1dt[0]
        dc1dt[0] = -3 * alpha * (c1[1] - c1_in) - beta * dq1dt[0]
        dc1dt[1] = -alpha * (2 * c1[2] + 3 * c1[1] - 6 * c1[0] + c1_in) - beta * dq1dt[1]
        dc1dt[NX_3rd - 1] = -3.0 * alpha * (3.0 * c1[NX_3rd - 1] - 4.0 * c1[NX_3rd - 2] + c1[NX_3rd - 3]) - beta * dq1dt[NX_3rd - 1]
        # dc2dt[0] = -alpha * (2 * c2[1] + 3 * c2[0] - 6 * c2_in + c2_in) - beta * dq2dt[0]
        dc2dt[0] = -3 * alpha * (c2[1] - c2_in) - beta * dq2dt[0]
        dc2dt[1] = -alpha * (2 * c2[2] + 3 * c2[1] - 6 * c2[0] + c2_in) - beta * dq2dt[1]
        dc2dt[NX_3rd - 1] = -3.0 * alpha * (3.0 * c2[NX_3rd - 1] - 4.0 * c2[NX_3rd - 2] + c2[NX_3rd - 3]) - beta * dq2dt[NX_3rd - 1]

    for i in range(2, NX_3rd - 1):
            dc1dt[i] = -alpha * (2 * c1[i + 1] + 3 * c1[i] - 6 * c1[i - 1] + c1[i - 2]) - beta * dq1dt[i]
            dc2dt[i] = -alpha * (2 * c2[i + 1] + 3 * c2[i] - 6 * c2[i - 1] + c2[i - 2]) - beta * dq2dt[i]

    return dxdt


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


@numba.jit("f8[:](f8,f8[:],f8[:],f8[:])", nopython=True)
def sim_model_flux_limiter(t, x, theta, params):
    H1, H2, K1, K2, b1, b2 = theta
    dx_flux, eb, um, Ca0, Cb0, tinj1, tinj2, tinj3, threshold = params

    v = um / eb
    alpha = 1 / dx_flux
    beta = (1 - eb) / eb

    c1 = x[0:NX_flux]
    q1 = x[NX_flux:NX_flux + NX_flux]
    c2 = x[2 * NX_flux:3 * NX_flux]
    q2 = x[3 * NX_flux:4 * NX_flux]
    dxdt = np.empty(4 * NX_flux)

    dc1dt = dxdt[0:NX_flux]
    dc2dt = dxdt[2 * NX_flux:3 * NX_flux]
    dq1dt = dxdt[NX_flux:NX_flux + NX_flux]
    dq2dt = dxdt[3 * NX_flux:4 * NX_flux]

    c1_in = Ca0 * (t <= tinj1 or (t >= tinj2 and t <= tinj3))
    c2_in = Cb0 * (t <= tinj1 or (t >= tinj2 and t <= tinj3))

    for i in range(0, NX_flux):
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
    dc1dt[NX_flux - 1] = alpha * (convective_flux(v, c1, NX_flux - 2, threshold) - v * c1[NX_flux - 1]) - beta * dq1dt[NX_flux - 1]
    dc2dt[NX_flux - 1] = alpha * (convective_flux(v, c2, NX_flux - 2, threshold) - v * c2[NX_flux - 1]) - beta * dq2dt[NX_flux - 1]

    for i in range(1, NX_flux - 1):
        dc1dt[i] = alpha * (convective_flux(v, c1, i - 1, threshold) - convective_flux(v, c1, i, threshold)) - beta * \
                   dq1dt[i]
        dc2dt[i] = alpha * (convective_flux(v, c2, i - 1, threshold) - convective_flux(v, c2, i, threshold)) - beta * \
                   dq2dt[i]

    return dxdt


def my_model_deconvoluted_FEM3rd(theta, t):
    nt = len(t)
    output = np.zeros((nt, 2))
    x0 = np.zeros(4 * NX_3rd)
    output[0, :] = 0
    theta_model = theta
    params = np.array([dx_3rd, eb, um, Ca0, Cb0, tinj1, tinj2, tinj3])

    sol = integrate.odeint(sim_model_3rdFEM, x0, t, args=(theta_model, params), tfirst=True)
    # sol1 = integrate.solve_ivp(sim_model,[0,t[-1]],x0,method="LSODA",t_eval=t,args=(H1,H2,Kapp1,Kapp2,dx,eb,um))
    # sol = sol1.y.T


    return np.vstack((sol[:, NX_3rd - 1], sol[:, 3 * NX_3rd - 1]))


def my_model_deconvoluted_flux_limiter(theta, t):
    nt = len(t)
    output = np.zeros(2 * nt)
    x0 = np.zeros(4*NX_flux)
    theta_model = theta
    params = np.array([dx_flux, eb, um, Ca0, Cb0, tinj1, tinj2, tinj3, threshold_FL])

    
    sol = integrate.odeint(sim_model_flux_limiter, x0, t, args=(theta_model, params), tfirst=True)
    # sol1 = integrate.solve_ivp(sim_model,[0,t[-1]],x0,method="LSODA",t_eval=t,args=(H1,H2,Kapp1,Kapp2,dx,eb,um))
    # sol = sol1.y.T
    output[:nt] = sol[:, NX_flux - 1]
    output[nt:] = sol[:, 3 * NX_flux - 1]

    return np.vstack([output[:nt], output[nt:]])


H1_true, H2_true = [3.23, 6.5]  # Henry's Constant [A, B]
k1_true, k2_true = [0.319, 0.19]  # Overall Mass Transfer Coefficient [A, B]
b1_true, b2_true = [0.21, 0.422]  # Affinity Coefficient [A, B]

theta_t = np.array([H1_true, H2_true, k1_true, k2_true, b1_true, b2_true])
t_list = np.linspace(0, 500, 2001)

# truemodel = my_model(theta_t, t_list)
truemodel = my_model_deconvoluted_FEM3rd(theta_t, t_list)

# sigma = 3.0e-6 # [g/L]
# sigma = 3.0e-5 # [g/L]
# sigma = 3.0e-4 # [g/L]
sigma = 0.01 * Ca0  # [g/L]

# make data
np.random.seed(7167425)  # set random seed, so the data is reproducible each time
ObservationError = sigma * np.random.standard_normal(truemodel.shape)
data = ObservationError + truemodel  # https://www.headboost.jp/numpy-random-standard-normal/


# define your really-complicated likelihood function that uses loads of external codes
def my_loglike(theta):
    """
    A Gaussian log-likelihood function for a model with parameters given in theta
    """
    N_sample = len(data)
    H1, H2, K1, K2, b1, b2 = theta
    # cal = my_model(np.array([H1,H2,k1,k2,K1,K2]), t_list)
    cal = my_model_deconvoluted_flux_limiter(np.array([H1, H2, K1, K2, b1, b2]), t_list)

    log_l = np.sum((data - cal) ** 2)

    # return -(0.5/sigma**2)*log_l
    # return -0.5*N_sample*np.log(2*np.pi*sigma**2) -(0.5/sigma**2)*log_l
    return log_l


ObjectiveVariable_0 = [Ha_initial, Hb_initial, ka_initial, kb_initial, ba_initial, bb_initial]  # 初期値
start_time = time.time()
results = minimize(my_loglike, ObjectiveVariable_0, method='nelder-mead', options={'fatol': 1.0e-4})
print(f"FEM simulation time [s]: {time.time()-start_time}")

print(results)
Result = my_model_deconvoluted_flux_limiter(results["x"], t_list)

print(f"初期値(Ha, Hb, ka, kb, ba, bb)：{ObjectiveVariable_0}")
print(f"カラム分割数N={NX_flux}")

plt.figure()
plt.scatter(t_list, data[0], color='mediumpurple', s=5, label="CA, sampledata")
plt.scatter(t_list, data[1], color='orange', s=5, label="CB, sampledata")
plt.plot(t_list, Result[0], color='blue', linewidth=2, label="CA, simulation")
plt.plot(t_list, Result[1], color='red', linewidth=2, label="CB, simulation")
plt.xlabel("Time[s]")
plt.ylabel("Concentration[g/L]")
plt.title("Fitting plot")
plt.legend()
plt.show()
plt.savefig(f"Results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=600)
