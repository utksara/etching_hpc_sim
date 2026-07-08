import numpy as np
from matplotlib import pyplot as plt 

def extract_from_image(image_path):
    # return points in Nx2 arr using simple filter based image processing
    pass

def etch_flux(point):
    x, y = point[0], point[1]
    y_comp = 0
    x_comp = 0.2/(1 + x**2)
    if abs(x) < 0.5:
        y_comp = 1/(1 + x**2)
    return np.array([x_comp, y_comp])

def normal(points, N):
    dx = np.ones(N) 
    dy = np.zeros(N) 
    dx[1:N-1] = (points[2:,0] - points[:N-2,0])
    dy[1:N-1] = (points[2:,1] - points[:N-2,1])
    dd = np.sqrt(dx**2 + dy**2)
    norm = np.zeros((N, 2))
    theta = np.arcsin(dy/dd) + np.pi/2
    norm[:,0] = np.cos(theta)
    norm[:,1] = np.sin(np.pi - theta)
    return norm
    
def derivative(points, N):
    derivative = np.zeros(N) 
    derivative[1:-1] = (points[2:][1] - points[:N-2][1])/(points[2:][0] - points[:N-2][0])

def get_plane_points(N, a):
    points = np.zeros((N,2))
    for i in range(0, N):
        x = (i - 50)/50
        y = a
        points[i][0] = x
        points[i][1] = y
    return points

def get_expo_points(N, a):
    points = np.zeros((N,2))
    for i in range(0, N):
        x = (i - 50)/50
        y = np.exp(-x**2/a)
        points[i][0] = x
        points[i][1] = y
    return points
  
N = 100
points = get_plane_points(N, 0.1)
p_norm = normal(points, N)

temporaral_data = []
dt = 1/100
for t in range(0, 100):
    for i in range(0, N):
        points[i,:] =  points[i,:] + etch_flux(points[i,:])*p_norm[i,:]*dt
    temporaral_data.append(points.copy())

# Set up the plot for animation
fig, ax = plt.subplots(figsize=(8, 6))
# Calculate limits based on all temporal data to keep the view stable
all_x = [p[:, 0] for p in temporaral_data]
all_y = [p[:, 1] for p in temporaral_data]
x_min, x_max = np.min(all_x), np.max(all_x)
y_min, y_max = np.min(all_y), np.max(all_y)

# Add margin to the plot limits
ax.set_xlim(x_min - 0.1, x_max + 0.1)
ax.set_ylim(y_min - 0.1, y_max + 0.1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Etching Simulation Animation")
ax.grid(True)

line, = ax.plot([], [], 'b-o', lw=2, markersize=3, label="Etch front")
ax.legend()

def init():
    line.set_data([], [])
    return line,

def update(frame):
    current_points = temporaral_data[frame]
    line.set_data(current_points[:, 0], current_points[:, 1])
    ax.set_title(f"Etching Simulation - Step {frame + 1}/100")
    return line,

# Create the animation
import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, update, frames=len(temporaral_data),
                              init_func=init, blit=True, interval=50)

# Save the animation as a GIF
output_gif = "etching_animation.gif"
try:
    ani.save(output_gif, writer='pillow', fps=20)
    print(f"Animation successfully saved to {output_gif}")
except Exception as e:
    print(f"Could not save animation: {e}")

plt.close(fig)