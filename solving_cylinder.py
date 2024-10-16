import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Parameters for the cylinder
radius = 1
height = 2
num_wedges = 10
angle_step = 360 / num_wedges

# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create and plot each wedge
for i in range(num_wedges):
    theta1 = np.deg2rad(i * angle_step)
    theta2 = np.deg2rad((i + 1) * angle_step)
    
    # Define the points for the wedge
    x = [0, radius * np.cos(theta1), radius * np.cos(theta2), 0]
    y = [0, radius * np.sin(theta1), radius * np.sin(theta2), 0]
    z_top = [height] * 4
    z_bottom = [0] * 4

    # Draw the top face of the wedge
    verts_top = [list(zip(x, y, z_top))]
    verts_bottom = [list(zip(x, y, z_bottom))]
    ax.add_collection3d(Poly3DCollection(verts_top, color='b', alpha=0.3, edgecolor='k'))
    ax.add_collection3d(Poly3DCollection(verts_bottom, color='b', alpha=0.3, edgecolor='k'))

    # Draw the sides of the wedge
    for j in range(4):
        x_side = [x[j], x[(j+1) % 4]]
        y_side = [y[j], y[(j+1) % 4]]
        z_side = [z_bottom[j], z_top[j]]
        ax.plot(x_side, y_side, z_side, color='k')

# Set plot limits and labels
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([0, height])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Cylinder Sliced into Ten 36-degree Wedges')

plt.show()
