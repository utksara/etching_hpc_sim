import os
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Path to the shared library
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "libetch.so"))

# Load the library
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Shared library not found at {lib_path}. Please run `make` first to compile it.")

lib = ctypes.CDLL(lib_path)

# Set argument types for:
# void run_simulation(float *points_in_out, float *trajectory_out, int N, int Nt, float dt)
lib.run_simulation.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # points_in_out (N x 2)
    ctypes.POINTER(ctypes.c_float),  # trajectory_out (Nt x N x 2)
    ctypes.c_int,                    # N
    ctypes.c_int,                    # Nt
    ctypes.c_float                   # dt
]
lib.run_simulation.restype = None

def get_plane_points(N, a):
    """
    Initialises N coordinates in Python.
    """
    points = np.zeros((N, 2), dtype=np.float32)
    points[0, 0] = -1.0
    points[0, 1] = a
    points[N - 1, 0] = 1.0
    points[N - 1, 1] = a
    
    points[1, 0] = -0.5 - 0.1
    points[1, 1] = a
    points[N - 2, 0] = 0.5 + 0.1
    points[N - 2, 1] = a
    
    for i in range(2, N - 2):
        points[i, 0] = (i - N / 2.0) / N
        points[i, 1] = a
    return points

def run_c_etch_sim(N=100, Nt=50, dt=0.01, a=0.1):
    """
    Sets up the arrays and invokes the C function run_simulation.
    Returns a NumPy array of shape (Nt, N, 2) containing the simulation trajectory.
    """
    # Initialize initial state
    points = get_plane_points(N, a)
    points_c = points.copy()
    
    # Pre-allocate trajectory buffer
    trajectory = np.zeros((Nt, N, 2), dtype=np.float32)
    
    # Extract ctypes pointers
    points_ptr = points_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    trajectory_ptr = trajectory.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    print(f"Calling C shared library: N = {N}, Nt = {Nt}, dt = {dt}")
    lib.run_simulation(points_ptr, trajectory_ptr, N, Nt, dt)
    print("Simulation finished. Data successfully stored in Python NumPy object.")
    
    return trajectory

def animate_trajectory(trajectory, filename="etching_animation_c.gif"):
    """
    Creates and saves an animation of the points trajectory.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    all_x = trajectory[:, :, 0]
    all_y = trajectory[:, :, 1]
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    ax.set_xlim(x_min - 0.1, x_max + 0.1)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Etching Simulation Animation (C Engine)")

    line, = ax.plot([], [], 'b-o', lw=2, markersize=3, label="Etch front")
    ax.legend()

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        current_points = trajectory[frame]
        line.set_data(current_points[:, 0], 1.0 - current_points[:, 1])
        ax.set_title(f"Etching Simulation - Step {frame + 1}/{len(trajectory)}")
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                  init_func=init, blit=True, interval=100)

    try:
        ani.save(filename, writer='pillow', fps=10)
        print(f"Animation successfully saved to {filename}")
    except Exception as e:
        print(f"Could not save animation: {e}")

    plt.close(fig)

if __name__ == "__main__":
    # Execute default run
    trajectory = run_c_etch_sim(N=100, Nt=50, dt=0.01, a=0.1)
    print("Trajectory shape:", trajectory.shape)
    
    # Animate and save
    animate_trajectory(trajectory)
