import deepxde as dde
import numpy as np
# Backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
import matplotlib.pyplot as plt
import random


L = 0.1  # Column length(m)
d = 0.0212  # Column diameter(m)
A = np.pi*d*d/4  # Column cross sectional area (m2)
e = 0.62  # porosity
v = 30.0/1000.0/1000/60.0/A/e  # Velocity (m/s)
Ha = 3.23  # Henry constant of A (-)
Hb = 6.5  # Henry constant of B (-)
f = 10  # Feed concentration (-)
ba = 0.21  # vol%
bb = 0.422  # vol%
te = 300  # final time (s)
t_inj = 100  # injected time (s)

# Normalization
scale_x = 100
scale_t = 1/30
scale_y = 1
L_scaled = L * scale_x
v_scaled = v * scale_x / scale_t
t_scaled = te * scale_t
t_inj_scaled = t_inj * scale_t


"""LDF model"""
DLa = 1.8e-7  # diffusion coefficient
DLb = 2.2e-7  # diffusion coefficient
ka = 0.64  # Mass transfer coefficient of A (1/s)
kb = 0.38  # Mass transfer coefficient of B (1/s)
DLa_scaled = DLa * scale_x**2 / scale_t
DLb_scaled = DLb * scale_x**2 / scale_t
ka_scaled = ka / scale_t
kb_scaled = kb / scale_t

# Hyperparameter Selection
random.seed(21365)

neuron = random.randint(10, 30)
Layer = random.randint(2, 5)
Lrate = random.choice((0.001, 0.0001, 0.00001))
Activation = random.choice(("relu", "tanh", "swish"))


def pde(x, y):
    y_star = scale_y * y
    ca, cb, qa, qb = y_star[:, 0:1], y_star[:, 1:2], y_star[:, 2:3], y_star[:, 3:]
    ca_x = dde.grad.jacobian(y_star, x, i=0, j=0)
    ca_t = dde.grad.jacobian(y_star, x, i=0, j=1)
    ca_xx = dde.grad.hessian(y_star, x, i=0, j=0, component=0)

    cb_x = dde.grad.jacobian(y_star, x, i=1, j=0)
    cb_t = dde.grad.jacobian(y_star, x, i=1, j=1)
    cb_xx = dde.grad.hessian(y_star, x, i=0, j=0, component=1)

    qa_t = dde.grad.jacobian(y_star, x, i=2, j=1)
    qb_t = dde.grad.jacobian(y_star, x, i=3, j=1)

    massbalance_liquid_a = (
            ca_t + (1-e)/e * qa_t - DLa_scaled * ca_xx + v_scaled * ca_x
    )
    massbalance_liquid_b = (
            cb_t + (1-e)/e * qb_t - DLb_scaled * cb_xx + v_scaled * cb_x
    )
    massbalance_solid_a = (
            ka_scaled * ((Ha * ca) / (1 + ba * ca + bb * cb) - qa) - qa_t
    )
    massbalance_solid_b = (
            kb_scaled * ((Hb * cb) / (1 + ba * ca + bb * cb) - qb) - qb_t
    )
    return [massbalance_liquid_a, massbalance_liquid_b, massbalance_solid_a, massbalance_solid_b]


def feed_concentration(x):
    # x, t = np.split(x, 2, axis=1)
    return np.piecewise(x[:, 1:], [x[:, 1:] <= t_inj_scaled, x[:, 1:] > t_inj_scaled], [lambda x: f, lambda x: 0])


def boundary_beg(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


geom = dde.geometry.Interval(0, L_scaled)
timedomain = dde.geometry.TimeDomain(0, t_scaled)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


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

"""Choose the Training points uniformly from the domain"""
z = np.linspace(0, (t_inj / 300) * L_scaled, num=300)
t = 0
Z, T = np.meshgrid(z, t)
Training_Points = np.vstack((Z.flatten(), T.flatten())).T

data = dde.data.TimePDE(
    geomtime,
    pde,
    [initial_condition_ca,
     initial_condition_cb,
     initial_condition_qa,
     initial_condition_qb,
     bc_beg_ca,
     bc_beg_cb],
    num_domain=10000,
    num_initial=300,
    num_boundary=300,
    anchors=Training_Points
)

layer_size = [2] + [neuron] * Layer + [4]
activation = Activation
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
# net.apply_output_transform(lambda x, y: y * scale_y)

model = dde.Model(data, net)
resampler = dde.callbacks.PDEResidualResampler(period=100)
early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-12, patience=10000)
model.compile("adam", lr=Lrate, loss_weights=[6, 15, 1e-2, 1e-2, 1, 1, 1, 1, 1, 1])
losshistory, train_state = model.train(iterations=500000, disregard_previous_best=True, callbacks=[resampler, early_stopping])


"""Get the domain: x = L_scaled and t from 0 to t_scaled"""
X_nn = L_scaled * np.ones((100, 1))
T_nn = np.linspace(0, t_scaled, 100).reshape(100, 1)
X_pred = np.append(X_nn, T_nn, axis=1)

y_pred = model.predict(X_pred)
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

"""Get the domain: x = 0.5*L_scaled and t from 0 to t_scaled"""
X_nn_h = 0.5*L_scaled * np.ones((100, 1))
T_nn_h = np.linspace(0, t_scaled, 100).reshape(100, 1)
X_pred_h = np.append(X_nn_h, T_nn_h, axis=1)

y_pred_h = model.predict(X_pred_h)
ca_pred_h, cb_pred_h, qa_pred_h, qb_pred_h = y_pred_h[:, 0:1], y_pred_h[:, 1:2], y_pred_h[:, 2:3], y_pred_h[:, 3:]
plt.figure()
plt.plot(T_nn_h / scale_t, ca_pred_h, color='blue', linewidth=3., label='Concentration A')
plt.plot(T_nn_h / scale_t, cb_pred_h, color='red', linewidth=3., label='Concentration B')
# plt.plot(X_pred, qa_pred)
# plt.plot(X_pred, qb_pred)

plt.legend()
plt.grid(True)
plt.xlabel('t')
plt.ylabel('Concentration')

plt.show()

"""Get the domain: x from 0 to L_scaled and t = t_scaled"""
X_nn_prime = np.linspace(0, L_scaled, 100).reshape(100, 1)
T_nn_prime = t_scaled * np.ones((100, 1))
X_pred_p = np.append(X_nn_prime, T_nn_prime, axis=1)

y_pred_p = model.predict(X_pred_p)
ca_pred_p, cb_pred_p, qa_pred_p, qb_pred_p = y_pred_p[:, 0:1], y_pred_p[:, 1:2], y_pred_p[:, 2:3], y_pred_p[:, 3:]
plt.figure()
plt.plot(X_nn_prime / scale_x, ca_pred_p, color='blue', linewidth=3., label='Concentration A')
plt.plot(X_nn_prime / scale_x, cb_pred_p, color='red', linewidth=3., label='Concentration B')
# plt.plot(X_pred, qa_pred)
# plt.plot(X_pred, qb_pred)

plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('Concentration')

plt.show()

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
print(neuron, Layer, Lrate, Activation)
