import numpy as np
from scipy.spatial import KDTree

def estimate_normals_2d(points: np.ndarray, k: int = 5, viewpoint: np.ndarray = None) -> np.ndarray:
    """
    Estimates 2D normal vectors for an unstructured set of points using Local PCA.
    
    Parameters:
        points (np.ndarray): An (N, 2) array of coordinates.
        k (int): Number of nearest neighbors to use for local covariance.
        viewpoint (np.ndarray): Optional (2,) coordinate used to consistently orient normals.
                                If provided, normals will be oriented pointing TOWARDS the viewpoint.
                                
    Returns:
        np.ndarray: An (N, 2) array of unit normal vectors.
    """
    N = points.shape[0]
    normals = np.zeros_like(points)
    
    # Build KD-Tree for efficient spatial query of k-nearest neighbors
    tree = KDTree(points)
    
    # Query k-nearest neighbors for all points
    # We query k+1 neighbors because the query point itself is included
    distances, indices = tree.query(points, k=k+1)
    
    for i in range(N):
        # Extract the local neighborhood coordinates (shape: (k+1, 2))
        neighbor_indices = indices[i]
        neighborhood = points[neighbor_indices]
        
        # Step 2: Compute centroid of local neighborhood (shape: (2,))
        centroid = np.mean(neighborhood, axis=0)
        
        # Step 3: Construct local covariance matrix (size: 2x2)
        # Shift neighborhood points relative to centroid
        deviations = neighborhood - centroid
        covariance_matrix = np.dot(deviations.T, deviations) / (k + 1)
        
        # Step 4: Solve eigenvalue problem
        # np.linalg.eigh is optimized for symmetric real matrices like covariance.
        # It conveniently returns eigenvalues in ascending order (eigenvalues[0] <= eigenvalues[1])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Step 5: Extract normal vector
        # The eigenvector corresponding to the smallest eigenvalue represents the direction
        # of least variance, which is perpendicular to the local curve (the normal vector).
        raw_normal = eigenvectors[:, 0]  # First column corresponds to eigenvalues[0]
        
        # Ensure it remains a unit vector (handles numerical edge cases)
        norm = np.linalg.norm(raw_normal)
        if norm > 1e-8:
            raw_normal /= norm
        else:
            raw_normal = np.array([0.0, 1.0]) # Fallback normal if neighborhood is singular
            
        # Step 6: Orientation Consistency
        if viewpoint is not None:
            # Vector pointing from the current point to the viewpoint
            to_viewpoint = viewpoint - points[i]
            # Ensure the normal points towards the viewpoint (dot product > 0)
            if np.dot(raw_normal, to_viewpoint) < 0:
                raw_normal = -raw_normal
                
        normals[i, :] = raw_normal
        
    return normals

if __name__ == "__main__":
    # Generate mock 2D points along a circle (perfectly structured curve)
    print("Generating circular mock point cloud...")
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    radius = 10.0
    circle_points = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    
    # Add a tiny bit of random noise
    np.random.seed(42)
    noisy_points = circle_points + np.random.normal(scale=0.1, size=circle_points.shape)
    
    # We define the origin (0, 0) as our viewpoint (inside the circle)
    # The outward-pointing normals should point away from (0, 0)
    # Therefore, let's set a viewpoint far outside the circle (e.g., at (0,0) to orient inwards,
    # or let's use the origin and orient outward by flipping the output if needed)
    origin_viewpoint = np.array([0.0, 0.0])
    
    # Estimate normals using k=4 neighbors
    # Orienting towards the origin viewpoint (inwards)
    normals_inward = estimate_normals_2d(noisy_points, k=4, viewpoint=origin_viewpoint)
    
    # Print results for a few points to verify
    print("\nCalculated Point Normals (oriented inwards towards [0,0]):")
    print(f"{'Point Coordinate':<25} | {'Unit Normal Vector':<25} | {'Cosine Similarity to Inward Radial'}")
    print("-" * 85)
    for i in range(5):
        pt = noisy_points[i]
        n = normals_inward[i]
        
        # Ideal inward radial direction
        ideal_dir = -pt / np.linalg.norm(pt)
        cosine_sim = np.dot(n, ideal_dir)
        
        print(f"({pt[0]:>6.2f}, {pt[1]:>6.2f})           | ({n[0]:>6.2f}, {n[1]:>6.2f})           | {cosine_sim:>6.4f}")