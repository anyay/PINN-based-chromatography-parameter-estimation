import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Example datasets
data1 = np.loadtxt('C:/Users/10716/OneDrive/桌面/Hyperparameter Selection contour data/FEM_forward_LDF.dat')
data2 = np.loadtxt('C:/Users/10716/OneDrive/桌面/Hyperparameter Selection contour data/21355 20 4 1e-05 relu/test.dat')

# Extract x, y, and z coordinates
x1, y1, z1 = data1[:, 0], data1[:, 1], data1[:, 2]
x2, y2, z2 = data2[:, 0], data2[:, 1], data2[:, 2]

# Interpolate z-values of dataset2 at x, y coordinates of dataset1
z2_at_dataset1 = griddata((x2, y2), z2, (x1, y1), method='cubic', fill_value=0.0)

# Interpolate z-values of dataset1 at x, y coordinates of dataset2
z1_at_dataset2 = griddata((x1, y1), z1, (x2, y2), method='cubic', fill_value=0.0)

# Compute differences
diff_at_dataset1 = z1 - z2_at_dataset1
diff_at_dataset2 = z1_at_dataset2 - z2

tolerance_FEM = 0.01  # Adjust the tolerance as needed
tolerance_PINN = 0.01  # Adjust the tolerance as needed
idx1 = np.abs(x1 - 10) < tolerance_FEM  # Indices where x1 is close to 10
idx2 = np.abs(x2 - 10) < tolerance_PINN  # Indices where x2 is close to 10

# Filtered data
y1_filtered, z1_filtered = y1[idx1], z1[idx1]
y2_filtered, z2_filtered = y2[idx2], z2[idx2]

plt.figure()

# Plotting data1
plt.plot(y1_filtered, z1_filtered, color='blue', label='FEM simulation')

# Plotting data2
plt.scatter(y2_filtered, z2_filtered, s=15, color='red', label='PINN simulation')

plt.xlabel('Normalized time')
plt.ylabel('Ca')
plt.title('Validation plot when x equals L')
plt.legend()
plt.show()
