import numpy as np
from matplotlib import pyplot as plt 
from scipy.optimize import minimize_scalar
from types import FunctionType
import matplotlib.animation as animation
from scipy.interpolate import make_interp_spline, CubicSpline

def to_angular_coordinates(points, centre, N):
    angular_coordinates = np.zeros((N, 2))
    angular_coordinates[0:2, 0] = 1
    angular_coordinates[0:2, 1] = abs(np.linalg.norm(points[0:2,:] - centre))
    angular_coordinates[N-2:N, 0] = -1
    angular_coordinates[N-2:N, 1] = abs(np.linalg.norm(points[N-2:N,:] - centre))
    
    for i in range(2, N-1):
        d = np.linalg.norm(points[i,:] - centre)
        angular_coordinates[i, 0] = np.arccos((points[i,0] - centre[0])/d)/(np.pi/2) - 1
        angular_coordinates[i, 1] = d
    return angular_coordinates

def to_cartesian_coordinates(points, centre, N):
    cartesian_coordinates = np.zeros((N, 2))
    
    # d = points[0,1]
    # cartesian_coordinates[0, :] = centre + d * np.array([1, 0])
    
    # d = points[1,1]
    # cartesian_coordinates[1, :] = centre + d * np.array([1, 0])
    
    # d = points[N-2,1]
    # cartesian_coordinates[N-2, :] = centre + d * np.array([-1, 0])
    
    # d = points[N-1,1]
    # cartesian_coordinates[N-1, :] = centre + d * np.array([-1, 0])
    
    for i in range(0, N):
        d = points[i,1]
        theta = (points[i,0] + 1) * np.pi/2
        print(np.array([round(np.cos(theta), 4), round(np.sin(theta), 4)]))
        cartesian_coordinates[i, :] = centre + d * np.array([round(np.cos(theta), 4), round(np.sin(theta), 4)])
    return cartesian_coordinates
    
def extract_from_image(image_path):
    # return points in Nx2 arr using simple filter based image processing
    pass

def get_etch_flux(point, a = 0.1):
    x, y = point[0], point[1]
    y_comp = 0
    x_comp = 0
    if abs(x) < 0.5 or y > a:
        y_comp = 1
    return np.array([x_comp, y_comp])

def get_normal(points, N):
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

def get_tangent(points, N):
    dx = np.ones(N) 
    dy = np.zeros(N) 
    dx[1:N-1] = (points[2:,0] - points[:N-2,0])
    dy[1:N-1] = (points[2:,1] - points[:N-2,1])
    dd = np.sqrt(dx**2 + dy**2)
    norm = np.zeros((N, 2))
    norm[:,0] = dx/dd
    norm[:,1] = dy/dd
    return norm
    
def derivative(points, N):
    derivative = np.zeros(N) 
    derivative[1:-1] = (points[2:][1] - points[:N-2][1])/(points[2:][0] - points[:N-2][0])

def get_plane_points(N, a):
    points = np.zeros((N,2))
    
    points[0,0] = -1
    points[0, 1] = a
    points[N - 1,0] = 1
    points[N - 1,1] = a
    
    points[1,0] = -0.5 - 0.1
    points[1, 1] = a
    points[N - 2,0] = 0.5 + 0.1
    points[N - 2,1] = a
    for i in range(2, N - 2):
        x = (i - N/2)/N
        y = a
        points[i,0] = x
        points[i,1] = y
    return points

def re_distribute_points(points, N):
    sub_points = points[1:N-1]
    dists = np.sqrt(np.sum(np.diff(sub_points, axis=0)**2, axis=1))
    s = np.concatenate(([0], np.cumsum(dists)))
    # task : instead of numpy interpolation used scipy cubic interpolation
    if s[-1] > 0:
        s_new = np.linspace(0, s[-1], N-2)
        new_x = np.interp(s_new, s, sub_points[:, 0])
        new_y = np.interp(s_new, s, sub_points[:, 1])
        points[1:N-1] = np.column_stack((new_x, new_y))
    return points

def get_expo_points(N, a):
    points = np.zeros((N,2))
    for i in range(0, N):
        x = (i - 50)/50
        y = np.exp(-x**2/a)
        points[i, 0] = x
        points[i, 1] = y
    return points

class UpdateMethod:
    def explicit_update(value, rate_function: FunctionType, time : float, time_step : float):
        return value + rate_function(value, time)*time_step
        
    def implicit_update(value, rate_function: FunctionType, time : float, time_step : float):
        initial_guess_1 = UpdateMethod.explicit_update(value, rate_function, time, time_step)
        # initial_guess_2 = UpdateMethod.explicit_update(initial_guess_1, rate_function, time_step)
        def objective(a):
            y_pred = UpdateMethod.explicit_update(a*value + (1 - a)*initial_guess_1, rate_function, time, time_step)
            return np.sum(abs((y_pred - value)/time_step  - rate_function(y_pred, time)))
        result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
        a = result.x
        return UpdateMethod.explicit_update(a*value + (1 - a)*initial_guess_1, rate_function, time, time_step)

def get_reaction_rate():
    pass

def track_surface(point_array, constraint_locus : np.ndarray):
    N = point_array.shape[0]
    for i in range(0, N):
        x_indx = constraint_locus
        point_array[i, 1] = min(constraint_locus[i, 1], point_array[i, 1]) 
    return point_array

def get_etch_rate(point_array, time):
    N = point_array.shape[0]
    p_norm = get_normal(point_array, N)
    removal_rate = np.zeros((N, 2))
    for i in range(0, N):
        # removal_rate[i, :] = np.linalg.norm(get_etch_flux(point_array[i, :])) * p_norm[i, :]
        removal_rate[i, :] = p_norm[i, :]
    removal_rate[[0, 1, N-2, N-1], :] = 0
    return removal_rate

def get_sputter_rate(point_array, time):
    N = point_array.shape[0]
    p_norm = get_normal(point_array, N)
    removal_rate = np.zeros((N, 2))
    for i in range(0, N):
        removal_rate[i, :] = np.inner(get_etch_flux(point_array[i, :]), p_norm[i, :]) * p_norm[i, :]
    removal_rate[[0, 1, N-2, N-1], :] = 0
    return removal_rate  


if __name__ == "__main__":
    N = 31
    points = get_plane_points(N, 0.1)
    p_norm = get_normal(points, N)

    temporaral_data = []
    Nt = 51
    dt = 1/Nt
    for t in range(0, Nt):
        points = UpdateMethod.implicit_update(points, get_etch_rate, t, dt)
        # points = track_surface(points)
        points = re_distribute_points(points, N)
        
        temporaral_data.append(points.copy())
        
    def animate(temporaral_data):
        # Set up the plot afor animation
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
        # ax.grid(True)

        line, = ax.plot([], [], 'b-o', lw=2, markersize=3, label="Etch front")
        ax.legend()

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            current_points = temporaral_data[frame]
            line.set_data(current_points[:, 0], 1 - current_points[:, 1])
            ax.set_title(f"Etching Simulation - Step {frame + 1}/100")
            return line,

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(temporaral_data),
                                    init_func=init, blit=True, interval=100)

        # Save the animation as a GIF
        output_gif = "etching_animation.gif"
        try:
            ani.save(output_gif, writer='pillow', fps=2)
            print(f"Animation successfully saved to {output_gif}")
        except Exception as e:
            print(f"Could not save animation: {e}")

        plt.close(fig)
        
    animate(temporaral_data)