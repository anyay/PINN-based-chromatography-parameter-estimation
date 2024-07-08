import numpy as np
import pandas as pd


data1 = pd.read_csv(r"C:\Users\10716\OneDrive\桌面\DeepXDE\DeepXDE\SMB\FEM_20s_1000DisPoints_A.csv")
data2 = pd.read_csv(r"C:\Users\10716\OneDrive\桌面\DeepXDE\DeepXDE\SMB\FEM_20s_1000DisPoints_B.csv")
np.save(r"C:\Users\10716\OneDrive\桌面\DeepXDE\DeepXDE\SMB\Inverse_A_20s_FC12_ForNoise_1000Dis points", data1)
np.save(r"C:\Users\10716\OneDrive\桌面\DeepXDE\DeepXDE\SMB\Inverse_B_20s_FC12_ForNoise_1000Dis points", data2)
traindata1 = np.load(r"C:\Users\10716\OneDrive\桌面\DeepXDE\DeepXDE\SMB\Inverse_A_20s_FC12_ForNoise_1000Dis points.npy")
traindata2 = np.load(r"C:\Users\10716\OneDrive\桌面\DeepXDE\DeepXDE\SMB\Inverse_B_20s_FC12_ForNoise_1000Dis points.npy")


xx1 = traindata1[1:, 2:3]
tt1 = traindata1[1:, 0:1]
Ca = traindata1[1:, 1:2]
xx2 = traindata2[1:, 2:3]
tt2 = traindata2[1:, 0:1]
Cb = traindata2[1:, 1:2]

sigma = 1.0e-1  # [g/L]
# make data
np.random.seed(7167425)  # set random seed, so the data is reproducible each time
ObservationError = sigma * np.random.standard_normal(Ca.shape)

Ca = ObservationError + Ca
Cb = ObservationError + Cb

Data_Ca = np.hstack((tt1, Ca, xx1))
Data_Cb = np.hstack((tt2, Cb, xx2))


df1 = pd.DataFrame(Data_Ca)
df2 = pd.DataFrame(Data_Cb)
df1.to_csv('Noise_A_20s_FC12_0.1_1000Dis point.csv')
df2.to_csv('Noise_B_20s_FC12_0.1_1000Dis point.csv')


