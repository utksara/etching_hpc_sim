import numpy as np
from scipy.spatial.distance import cdist
from fastdtw import fastdtw
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt

def calculate_curve_similarity(curve1, curve2):
    """
    Calculates the similarity between two 2D curves of different sizes 
    using Fast Dynamic Time Warping (FastDTW).
    
    Parameters:
    curve1 (numpy.ndarray): Nx2 array of coordinates
    curve2 (numpy.ndarray): Mx2 array of coordinates
    
    Returns:
    float: The DTW distance (lower means more similar, 0.0 is identical)
    list: The optimal warping path matching indices between the two curves
    """
    # Ensure inputs are numpy arrays
    c1 = np.asarray(curve1, dtype=float)
    c2 = np.asarray(curve2, dtype=float)
    
    # Optional but highly recommended: Normalize or center your curves if 
    # you care about shape similarity regardless of translation/scale.
    c1 -= np.mean(c1, axis=0)
    c2 -= np.mean(c2, axis=0)
    
    # Compute FastDTW distance using Euclidean distance as the metric
    distance, path = fastdtw(c1, c2, dist=2)
    return distance/len(path)

def test_linear_interp():
    N = 10
    x = np.linspace(-1, 1, N)
    x [0] = -2
    x [-1] = 1.5
    y = x**2
    x_interp = np.linspace(-2, 2, N*10)
    plt.plot(x, y, label="actual")
    cs = CubicSpline(x, y)
    y_interp = cs(x_interp)
    plt.plot(x_interp, y_interp, label="scipy")
    d = calculate_curve_similarity(y, y_interp)
    print(f"d scipy interp {d}")    
    
    y_interp = np.interp(x_interp, x, y)
    plt.plot(x_interp, y_interp, label="numpy")
    d = calculate_curve_similarity(y, y_interp)
    print(f"d np.interp {d}")
    plt.legend()
    plt.show()
    assert d <= 0.1
    
def test_uniform_arc_length_reinterpolation():
    # Create points of a curve where x is not strictly increasing/decreasing (e.g. a semi-circle)
    theta = np.linspace(0, np.pi, 20)
    points = np.column_stack((np.cos(theta), np.sin(theta)))
    
    # Calculate cumulative distance along the curve
    dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    s = np.concatenate(([0], np.cumsum(dists)))
    
    s_new = np.linspace(0, s[-1], 20)
    new_x = np.interp(s_new, s, points[:, 0])
    new_y = np.interp(s_new, s, points[:, 1])
    reinterp_points = np.column_stack((new_x, new_y))
    
    # Since they are redistributed uniformly by arc-length, the distance between consecutive points should be nearly equal
    new_dists = np.sqrt(np.sum(np.diff(reinterp_points, axis=0)**2, axis=1))
    assert np.std(new_dists) < 1e-4
