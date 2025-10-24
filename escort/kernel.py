import warnings

from numba import jit, prange
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


# & Numba JIT-compiled functions for maximum performance
@jit(nopython=True, parallel=True, fastmath=True)
def compute_squared_distances_numba(x, y):
    """
        JIT-compiled pairwise squared distance computation.
    """
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    dists = np.zeros((n, m))
    
    for i in prange(n):
        for j in range(m):
            dist = 0.0
            for k in range(d):
                diff = x[i, k] - y[j, k]
                dist += diff * diff
            dists[i, j] = dist
    
    return dists


@jit(nopython=True, parallel=True, fastmath=True)
def compute_rbf_kernel_numba(squared_dists, bandwidth):
    """
        JIT-compiled RBF kernel evaluation.
    """
    n, m = squared_dists.shape
    K = np.zeros((n, m))
    
    for i in prange(n):
        for j in range(m):
            arg = -squared_dists[i, j] / (2 * bandwidth)
            # Clip to prevent overflow
            if arg < -700:
                K[i, j] = 0.0
            else:
                K[i, j] = np.exp(arg)
    
    return K


@jit(nopython=True, parallel=True, fastmath=True)
def compute_rbf_gradient_numba(x, y, K, bandwidth):
    """
        JIT-compiled RBF gradient computation.
    """
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    grad_K = np.zeros((n, m, d))
    
    for i in prange(n):
        for j in range(m):
            for k in range(d):
                diff = x[i, k] - y[j, k]
                grad_K[i, j, k] = K[i, j] * (-diff / bandwidth)
    
    return grad_K


class ImprovedRBFKernel:
    """
        Improved RBF kernel with better numerical stability and adaptive bandwidth.
    """
    def __init__(self, bandwidth=1.0, adaptive=True, bandwidth_scale=0.5, 
                min_bandwidth=1e-5, max_bandwidth=100.0, use_numba=True):
        """
            Initialize the improved RBF kernel.
            
            Args:
                bandwidth: Initial kernel bandwidth
                adaptive: Whether to adapt bandwidth to median distances
                bandwidth_scale: Scaling factor for adaptive bandwidth
                min_bandwidth: Minimum allowed bandwidth value
                max_bandwidth: Maximum allowed bandwidth value
                use_numba: Whether to use Numba JIT compilation for speed
        """
        # & Store the initial/fixed bandwidth value
        self.bandwidth = bandwidth
        
        # & Flag to determine if bandwidth should adapt to data
        self.adaptive = adaptive
        
        # & Scaling factor applied to median distance when computing adaptive bandwidth
        self.bandwidth_scale = bandwidth_scale
        
        # & Lower bound for bandwidth to prevent numerical issues with very small values
        self.min_bandwidth = min_bandwidth
        
        # & Upper bound for bandwidth to prevent kernel from becoming too flat
        self.max_bandwidth = max_bandwidth
        
        # & Storage for computed adaptive bandwidth (initially None)
        self.adaptive_bandwidth = None
        
        # Flag to use Numba acceleration
        self.use_numba = use_numba


    def _compute_pairwise_distances(self, x, y=None):
        """
            Compute pairwise squared distances with numerical safeguards.
            
            Args:
                x: First set of points of shape (n, d)
                y: Second set of points of shape (m, d), defaults to x
                
            Returns:
                Pairwise squared distances of shape (n, m)
        """
        # & Replace any NaN or Inf values in x with safe values
        x = np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # & Check if we're computing distances between two different sets
        if y is not None:
            # & Clean y similarly
            y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # & Initialize distance matrix for x vs y
            try:
                dists = cdist(x, y, 'sqeuclidean')
            except:
                if self.use_numba:
                    # & Use JIT-compiled version for speed
                    dists = compute_squared_distances_numba(x, y)
                else:
                    # & Compute distance between each pair of points
                    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
                    
                    # & Check if any component is very large (potential overflow)
                    max_abs_diff = np.max(np.abs(diff), axis=2)
                    large_diff_mask = max_abs_diff > 1e5
                    
                    # & For normal ranges, use standard squared Euclidean distance
                    dists = np.sum(diff * diff, axis=2)
                    
                    # & For very large differences, use log-sum-exp trick to avoid overflow
                    if np.any(large_diff_mask):
                        for i, j in zip(*np.where(large_diff_mask)):
                            # & For very large differences, use log-sum-exp trick to avoid overflow
                            log_diffs = np.log(np.abs(diff[i, j]) + 1e-10)
                            
                            # & Find maximum log value for normalization
                            max_log = np.max(log_diffs)
                            
                            # & sq_dist = exp(2*max_log) * sum(exp(2*(log_diffs - max_log)))
                            dists[i, j] = np.exp(max_log*2) * np.sum(np.exp(2*(log_diffs - max_log)))
        else:
            # & Computing distances within the same set (x vs x)
            try:
                # & pdist computes upper triangular part, squareform makes it symmetric
                pairwise = squareform(pdist(x, 'sqeuclidean'))
                
                # & Convert to float64 for numerical precision
                dists = np.array(pairwise, dtype=np.float64)
            except Exception:
                # & If scipy fails, fall back to manual computation
                if self.use_numba:
                    # & Use JIT-compiled version
                    dists = compute_squared_distances_numba(x, x)
                else:
                    # & Initialize symmetric distance matrix
                    # & Compute difference vector
                    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
                    
                    # & Compute squared Euclidean distance
                    dists = np.sum(diff * diff, axis=2)
        
        # & Ensure all distances are non-negative (fix numerical errors)
        dists = np.maximum(0.0, dists)
        
        # & Cap maximum distance to prevent numerical issues
        dists = np.minimum(dists, 1e10)
        
        return dists
    

    def evaluate(self, x, y=None):
        """
            Evaluate the kernel matrix K(x, y).
            
            Args:
                x: First set of points of shape (n, d)
                y: Second set of points of shape (m, d), defaults to x
                
            Returns:
                Kernel matrix of shape (n, m)
        """
        # & First compute all pairwise squared distances
        squared_dists = self._compute_pairwise_distances(x, y)
        
        # & Check if we should adapt bandwidth based on data
        if self.adaptive:
            # & Only adapt when computing K(x,x), not K(x,y)
            if y is None:
                # & Extract non-diagonal elements (diagonal is always 0 for x vs x)
                # & Create boolean mask for off-diagonal elements
                distances = squared_dists[~np.eye(squared_dists.shape[0], dtype=bool)]
                
                # & Check if we have any valid distances
                if len(distances) > 0:
                    # & Compute median of actual distances (not squared)
                    med_dist = np.median(np.sqrt(distances))
                    
                    # & Scale the median distance by our scaling factor
                    self.adaptive_bandwidth = med_dist * self.bandwidth_scale
                    
                    # & Higher dimensions need larger bandwidth to combat kernel degeneracy
                    dim = x.shape[1]
                    dim_factor = np.sqrt(dim) / 2
                    self.adaptive_bandwidth *= dim_factor
                    
                    # & Ensure bandwidth stays within reasonable bounds
                    self.adaptive_bandwidth = np.clip(
                        self.adaptive_bandwidth, 
                        self.min_bandwidth, 
                        self.max_bandwidth
                    )
                else:
                    # & If all particles are identical, use default bandwidth
                    self.adaptive_bandwidth = 1.0
            else:
                # & When computing K(x,y), check if we have a previously computed adaptive bandwidth
                if self.adaptive_bandwidth is None:
                    # & No adaptive bandwidth computed yet, use reasonable default
                    self.adaptive_bandwidth = 1.0
        
        # & Select which bandwidth to use
        bandwidth = self.adaptive_bandwidth if self.adaptive and self.adaptive_bandwidth is not None else self.bandwidth
        
        # & Compute RBF kernel: K(x,y) = exp(-||x-y||^2 / (2*bandwidth))
        if self.use_numba and squared_dists.size > 500:  # & Use Numba for larger arrays
            K = compute_rbf_kernel_numba(squared_dists, bandwidth)
        else:
            K = np.exp(-squared_dists / (2 * bandwidth))
        
        # & Check for numerical issues in kernel matrix
        if not np.all(np.isfinite(K)):
            # & Warn about numerical issues
            warnings.warn("Kernel matrix contains NaN or Inf. Using fallback computation.")
            
            # & Clip distances before exponentiating to prevent overflow
            K = np.exp(-np.minimum(squared_dists, 1e6) / (2 * bandwidth))
            
            # & Replace any remaining NaN or Inf values with safe values
            K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        
        return K
    

    def gradient(self, x, y=None):
        """
            Compute gradient of the kernel function with respect to x.
            
            Args:
                x: First set of points of shape (n, d)
                y: Second set of points of shape (m, d), defaults to x
                
            Returns:
                Gradient of the kernel with shape (n, m, d) if y is provided,
                otherwise shape (n, n, d)
        """
        # & If y not provided, compute gradient of K(x,x) w.r.t. x
        if y is None:
            y = x
        
        # & Get dimensions
        n, d = x.shape  # & n particles, d dimensions
        m = y.shape[0]  # & m reference points
        
        # & Initialize gradient tensor
        # & grad_K[i,j,k] = derivative of K(x_i, y_j) w.r.t. x_i[k]
        
        # & First compute kernel values
        K = self.evaluate(x, y)
        
        # & Get the bandwidth that was used in kernel computation
        bandwidth = self.adaptive_bandwidth if self.adaptive and self.adaptive_bandwidth is not None else self.bandwidth
        
        # & Compute gradient for each pair of points
        if self.use_numba and K.size > 500:  # & Use Numba for larger arrays
            grad_K = compute_rbf_gradient_numba(x, y, K, bandwidth)
        else:
            # & Compute difference vector (x_i - y_j)
            diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
            
            # & Gradient of RBF kernel: ∇_x K(x,y) = K(x,y) * (-(x-y)/bandwidth)
            grad_K = K[:, :, np.newaxis] * (-diff / bandwidth)
        
        # & Check for numerical issues in gradient
        if not np.all(np.isfinite(grad_K)):
            # & Warn about numerical issues
            warnings.warn("Kernel gradient contains NaN or Inf. Using fallback computation.")
            
            # & Replace problematic values with zeros
            grad_K = np.nan_to_num(grad_K, nan=0.0, posinf=0.0, neginf=0.0)
        
        return grad_K
