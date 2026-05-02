import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from typing import Callable, Dict, List, Tuple, Optional
import warnings


# ============================================================================
# Visualization Utilities
# ============================================================================

def visualize_particle_belief(particles: np.ndarray,
                            mode_stats: Optional[Dict] = None,
                            true_state: Optional[np.ndarray] = None,
                            title: str = "Particle Belief",
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (10, 8)):
    """
    Visualize particle-based belief distribution.
    
    Args:
        particles: Particle states (n_particles, state_dim)
        mode_stats: Dictionary with mode statistics from compute_mode_statistics
        true_state: Optional true state to highlight
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    n_particles, state_dim = particles.shape
    
    if state_dim > 3:
        warnings.warn(f"Can only visualize up to 3D. Showing first 3 dimensions of {state_dim}D state.")
        particles = particles[:, :3]
        state_dim = 3
    
    fig = plt.figure(figsize=figsize)
    
    if state_dim == 1:
        # 1D visualization
        ax = fig.add_subplot(111)
        ax.hist(particles[:, 0], bins=30, alpha=0.7, density=True, color='blue')
        ax.scatter(particles[:, 0], np.zeros_like(particles[:, 0]), alpha=0.3, s=20, c='red')
        
        if true_state is not None:
            ax.axvline(true_state[0], color='green', linestyle='--', linewidth=2, label='True State')
        
        ax.set_xlabel('State')
        ax.set_ylabel('Density')
        
    elif state_dim == 2:
        # 2D visualization
        ax = fig.add_subplot(111)
        
        # Plot particles colored by mode if available
        if mode_stats is not None and 'labels' in mode_stats:
            labels = mode_stats['labels']
            ax.scatter(particles[:, 0], particles[:, 1],
                            c=labels, cmap='viridis', alpha=0.6, s=30)
            
            # Plot mode centers
            centers = mode_stats['mode_centers']
            ax.scatter(centers[:, 0], centers[:, 1], 
                    c='red', marker='x', s=200, linewidths=3, label='Mode Centers')
            
            # Draw ellipses for covariances
            for i, (center, cov) in enumerate(zip(centers, mode_stats['mode_covariances'])):
                plot_covariance_ellipse(ax, center[:2], cov[:2, :2], alpha=0.3)
        else:
            ax.scatter(particles[:, 0], particles[:, 1], alpha=0.6, s=30)
        
        if true_state is not None:
            ax.scatter(true_state[0], true_state[1], 
                    color='green', marker='*', s=200, label='True State')
        
        ax.set_xlabel('State Dim 1')
        ax.set_ylabel('State Dim 2')
        
    else:  # state_dim == 3
        # 3D visualization
        ax = fig.add_subplot(111, projection='3d')
        
        if mode_stats is not None and 'labels' in mode_stats:
            labels = mode_stats['labels']
            ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2],
                            c=labels, cmap='viridis', alpha=0.6, s=30)
            
            # Plot mode centers
            centers = mode_stats['mode_centers']
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                    c='red', marker='x', s=200, label='Mode Centers')
        else:
            ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], alpha=0.6, s=30)
        
        if true_state is not None:
            ax.scatter(true_state[0], true_state[1], true_state[2],
                    color='green', marker='*', s=200, label='True State')
        
        ax.set_xlabel('State Dim 1')
        ax.set_ylabel('State Dim 2')
        ax.set_zlabel('State Dim 3')
    
    ax.set_title(title)
    if true_state is not None or (mode_stats is not None and 'mode_centers' in mode_stats):
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_covariance_ellipse(ax, mean: np.ndarray, cov: np.ndarray, 
                          n_std: float = 2.0, alpha: float = 0.3, **kwargs):
    """
    Plot covariance ellipse for 2D Gaussian.
    
    Args:
        ax: Matplotlib axis
        mean: 2D mean
        cov: 2x2 covariance matrix
        n_std: Number of standard deviations
        alpha: Transparency
        **kwargs: Additional plot arguments
    """
    from matplotlib.patches import Ellipse
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Compute angle
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # Compute width and height
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    
    # Create ellipse
    ellipse = Ellipse(mean, width, height, angle=angle, alpha=alpha, **kwargs)
    ax.add_patch(ellipse)


def plot_belief_evolution(belief_history: List[np.ndarray],
                        true_states: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None):
    """
    Plot evolution of belief over time.
    
    Args:
        belief_history: List of particle arrays over time
        true_states: Optional true state trajectory
        save_path: Path to save animation/figure
    """
    if len(belief_history) == 0:
        warnings.warn("No belief history to plot")
        return
    
    n_steps = len(belief_history)
    state_dim = belief_history[0].shape[1]
    
    if state_dim > 2:
        warnings.warn(f"Plotting first 2 dimensions of {state_dim}D state")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot particle trajectories
    for t in range(n_steps):
        particles = belief_history[t]
        alpha = (t + 1) / n_steps  # Fade in over time
        ax.scatter(particles[:, 0], particles[:, 1], 
                  alpha=alpha * 0.3, s=10, c=f'C{t % 10}')
    
    # Plot true state trajectory if available
    if true_states is not None:
        ax.plot(true_states[:, 0], true_states[:, 1], 
            'g-', linewidth=2, label='True Trajectory')
        ax.scatter(true_states[0, 0], true_states[0, 1], 
                color='green', marker='o', s=100, label='Start')
        ax.scatter(true_states[-1, 0], true_states[-1, 1], 
                color='red', marker='*', s=100, label='End')
    
    ax.set_xlabel('State Dim 1')
    ax.set_ylabel('State Dim 2')
    ax.set_title('Belief Evolution Over Time')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Metric Computation Utilities
# ============================================================================

def compute_mmd(samples1: np.ndarray, samples2: np.ndarray, 
                kernel_bandwidth: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy between two sets of samples.
    
    Args:
        samples1: First set of samples
        samples2: Second set of samples
        kernel_bandwidth: RBF kernel bandwidth
        
    Returns:
        MMD value
    """
    n1, n2 = len(samples1), len(samples2)
    
    # Compute kernel matrices
    K11 = rbf_kernel(samples1, samples1, kernel_bandwidth)
    K22 = rbf_kernel(samples2, samples2, kernel_bandwidth)
    K12 = rbf_kernel(samples1, samples2, kernel_bandwidth)
    
    # Compute MMD
    mmd = np.sqrt(
        K11.sum() / (n1 * n1) + 
        K22.sum() / (n2 * n2) - 
        2 * K12.sum() / (n1 * n2)
    )
    
    return mmd


def rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Compute RBF kernel matrix.
    
    Args:
        X: First set of points
        Y: Second set of points
        bandwidth: Kernel bandwidth
        
    Returns:
        Kernel matrix
    """
    pairwise_dists = cdist(X, Y, 'sqeuclidean')
    K = np.exp(-pairwise_dists / (2 * bandwidth))
    return K


def compute_wasserstein_distance(samples1: np.ndarray, samples2: np.ndarray) -> float:
    """
    Compute 1-Wasserstein distance between two sets of samples.
    
    Args:
        samples1: First set of samples
        samples2: Second set of samples
        
    Returns:
        Wasserstein distance
    """
    from scipy.stats import wasserstein_distance
    
    if samples1.shape[1] == 1:
        # 1D case - use scipy
        return wasserstein_distance(samples1[:, 0], samples2[:, 0])
    else:
        # Multi-dimensional case - use linear assignment
        from scipy.optimize import linear_sum_assignment
        
        # Compute pairwise distances
        distances = cdist(samples1, samples2)
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(distances)
        
        # Compute average distance
        return distances[row_ind, col_ind].mean()


def compute_sliced_wasserstein(samples1: np.ndarray, samples2: np.ndarray,
                            n_projections: int = 100) -> float:
    """
    Compute Sliced Wasserstein distance.
    
    Args:
        samples1: First set of samples
        samples2: Second set of samples
        n_projections: Number of random projections
        
    Returns:
        Sliced Wasserstein distance
    """
    dim = samples1.shape[1]
    distances = []
    
    for _ in range(n_projections):
        # Random projection direction
        theta = np.random.randn(dim)
        theta = theta / np.linalg.norm(theta)
        
        # Project samples
        proj1 = samples1 @ theta
        proj2 = samples2 @ theta
        
        # Sort projections
        proj1_sorted = np.sort(proj1)
        proj2_sorted = np.sort(proj2)
        
        # Compute 1D Wasserstein
        dist = np.mean(np.abs(proj1_sorted - proj2_sorted))
        distances.append(dist)
    
    return np.mean(distances)


def compute_correlation_error(particles: np.ndarray, 
                            true_correlation: Optional[np.ndarray] = None) -> float:
    """
    Compute error in correlation structure.
    
    Args:
        particles: Particle samples
        true_correlation: True correlation matrix (if known)
        
    Returns:
        Correlation error (Frobenius norm)
    """
    # Compute empirical correlation
    emp_correlation = np.corrcoef(particles.T)
    
    if true_correlation is None:
        # If no true correlation given, return condition number as proxy
        return np.linalg.cond(emp_correlation)
    else:
        # Compute Frobenius norm of difference
        return np.linalg.norm(emp_correlation - true_correlation, 'fro')


def compute_mode_coverage(particles: np.ndarray, 
                        true_modes: List[np.ndarray],
                        threshold: float = 0.5) -> float:
    """
    Compute fraction of true modes covered by particles.
    
    Args:
        particles: Particle samples
        true_modes: List of true mode centers
        threshold: Distance threshold for coverage
        
    Returns:
        Mode coverage ratio (0-1)
    """
    covered_modes = 0
    
    for mode in true_modes:
        # Check if any particle is close to this mode
        distances = np.linalg.norm(particles - mode, axis=1)
        if np.min(distances) < threshold:
            covered_modes += 1
    
    return covered_modes / len(true_modes)


# ============================================================================
# Mode Analysis Utilities
# ============================================================================

def detect_modes_meanshift(particles: np.ndarray, 
                        bandwidth: Optional[float] = None) -> Dict:
    """
    Detect modes using mean shift clustering.
    
    Args:
        particles: Particle samples
        bandwidth: Kernel bandwidth for mean shift
        
    Returns:
        Dictionary with mode information
    """
    from sklearn.cluster import MeanShift
    
    if bandwidth is None:
        # Estimate bandwidth using quantile
        from sklearn.cluster import estimate_bandwidth
        bandwidth = estimate_bandwidth(particles, quantile=0.3)
    
    # Apply mean shift
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(particles)
    
    # Get results
    labels = ms.labels_
    centers = ms.cluster_centers_
    n_modes = len(centers)
    
    # Compute mode weights
    unique_labels, counts = np.unique(labels, return_counts=True)
    weights = counts / len(particles)
    
    return {
        'num_modes': n_modes,
        'mode_centers': centers,
        'mode_weights': weights,
        'labels': labels,
        'bandwidth': bandwidth
    }


def compute_belief_entropy(particles: np.ndarray, 
                        bandwidth: Optional[float] = None) -> float:
    """
    Estimate entropy of particle-based belief.
    
    Args:
        particles: Particle samples
        bandwidth: Kernel bandwidth
        
    Returns:
        Entropy estimate
    """
    n_particles, dim = particles.shape
    
    if bandwidth is None:
        # Scott's rule
        bandwidth = n_particles**(-1./(dim+4))
    
    # Kernel density estimation
    log_densities = []
    for i in range(n_particles):
        # Leave-one-out density estimation
        others = np.concatenate([particles[:i], particles[i+1:]])
        distances = np.linalg.norm(others - particles[i], axis=1)
        kernel_vals = np.exp(-0.5 * (distances / bandwidth)**2)
        density = kernel_vals.mean() / ((2*np.pi)**0.5 * bandwidth)**dim
        log_densities.append(np.log(density + 1e-10))
    
    # Estimate entropy
    entropy = -np.mean(log_densities)
    return entropy


def analyze_belief_multimodality(particles: np.ndarray, 
                            n_neighbors: int = 5) -> Dict:
    """
    Analyze multi-modal structure of belief.
    
    Args:
        particles: Particle samples
        n_neighbors: Number of neighbors for local density
        
    Returns:
        Dictionary with multimodality analysis
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Compute local densities
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(particles)
    distances, _ = nbrs.kneighbors(particles)
    local_densities = 1.0 / (distances.mean(axis=1) + 1e-10)
    
    # Find local maxima (potential modes)
    is_local_max = np.zeros(len(particles), dtype=bool)
    for i in range(len(particles)):
        neighbor_indices = nbrs.kneighbors([particles[i]], return_distance=False)[0]
        if local_densities[i] >= local_densities[neighbor_indices].max():
            is_local_max[i] = True
    
    n_potential_modes = is_local_max.sum()
    
    # Compute separability (average inter-mode distance)
    if n_potential_modes > 1:
        mode_particles = particles[is_local_max]
        inter_mode_distances = cdist(mode_particles, mode_particles)
        np.fill_diagonal(inter_mode_distances, np.inf)
        separability = inter_mode_distances.min(axis=1).mean()
    else:
        separability = 0.0
    
    return {
        'n_potential_modes': n_potential_modes,
        'mode_separability': separability,
        'density_variance': np.var(local_densities),
        'max_density': local_densities.max(),
        'min_density': local_densities.min()
    }


# ============================================================================
# Performance Evaluation Utilities
# ============================================================================

def evaluate_belief_quality(particles: np.ndarray,
                        true_distribution: Optional[Callable] = None,
                        true_samples: Optional[np.ndarray] = None) -> Dict:
    """
    Comprehensive evaluation of belief quality.
    
    Args:
        particles: Particle samples
        true_distribution: True distribution function (optional)
        true_samples: Samples from true distribution (optional)
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['mean'] = particles.mean(axis=0)
    metrics['std'] = particles.std(axis=0)
    metrics['entropy'] = compute_belief_entropy(particles)
    
    # Mode analysis
    mode_stats = detect_modes_meanshift(particles)
    metrics['num_modes'] = mode_stats['num_modes']
    
    # Multi-modality analysis
    multimodal_stats = analyze_belief_multimodality(particles)
    metrics.update(multimodal_stats)
    
    # If true samples available, compute distances
    if true_samples is not None:
        metrics['mmd'] = compute_mmd(particles, true_samples)
        metrics['wasserstein'] = compute_wasserstein_distance(particles, true_samples)
        metrics['sliced_wasserstein'] = compute_sliced_wasserstein(particles, true_samples)
    
    # If true distribution available, compute likelihood
    if true_distribution is not None:
        log_likelihoods = [true_distribution(p) for p in particles]
        metrics['mean_log_likelihood'] = np.mean(log_likelihoods)
    
    return metrics


def compute_tracking_error(particle_means: np.ndarray,
                        true_states: np.ndarray) -> Dict:
    """
    Compute tracking error metrics.
    
    Args:
        particle_means: Mean of particles at each timestep
        true_states: True states at each timestep
        
    Returns:
        Dictionary with error metrics
    """
    errors = np.linalg.norm(particle_means - true_states, axis=1)
    
    return {
        'rmse': np.sqrt(np.mean(errors**2)),
        'mae': np.mean(errors),
        'max_error': np.max(errors),
        'final_error': errors[-1]
    }


# ============================================================================
# Data Processing Utilities
# ============================================================================

def normalize_particles(particles: np.ndarray,
                    method: str = 'standard') -> Tuple[np.ndarray, Dict]:
    """
    Normalize particles for stable computation.
    
    Args:
        particles: Particle samples
        method: Normalization method ('standard', 'minmax', 'robust')
        
    Returns:
        Normalized particles and normalization parameters
    """
    if method == 'standard':
        mean = particles.mean(axis=0)
        std = particles.std(axis=0) + 1e-8
        normalized = (particles - mean) / std
        params = {'mean': mean, 'std': std}
        
    elif method == 'minmax':
        min_val = particles.min(axis=0)
        max_val = particles.max(axis=0)
        range_val = max_val - min_val + 1e-8
        normalized = (particles - min_val) / range_val
        params = {'min': min_val, 'max': max_val}
        
    elif method == 'robust':
        median = np.median(particles, axis=0)
        mad = np.median(np.abs(particles - median), axis=0) + 1e-8
        normalized = (particles - median) / mad
        params = {'median': median, 'mad': mad}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_particles(normalized: np.ndarray,
                        params: Dict,
                        method: str = 'standard') -> np.ndarray:
    """
    Denormalize particles back to original scale.
    
    Args:
        normalized: Normalized particles
        params: Normalization parameters
        method: Normalization method used
        
    Returns:
        Denormalized particles
    """
    if method == 'standard':
        return normalized * params['std'] + params['mean']
    elif method == 'minmax':
        return normalized * (params['max'] - params['min']) + params['min']
    elif method == 'robust':
        return normalized * params['mad'] + params['median']
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resample_particles(particles: np.ndarray,
                    weights: Optional[np.ndarray] = None,
                    n_samples: Optional[int] = None,
                    method: str = 'systematic') -> np.ndarray:
    """
    Resample particles according to weights.
    
    Args:
        particles: Particle samples
        weights: Particle weights (uniform if None)
        n_samples: Number of samples (same as input if None)
        method: Resampling method ('systematic', 'multinomial', 'stratified')
        
    Returns:
        Resampled particles
    """
    n_particles = len(particles)
    
    if weights is None:
        weights = np.ones(n_particles) / n_particles
    else:
        weights = weights / weights.sum()
    
    if n_samples is None:
        n_samples = n_particles
    
    if method == 'systematic':
        # Systematic resampling
        positions = (np.arange(n_samples) + np.random.uniform()) / n_samples
        cumsum = np.cumsum(weights)
        indices = np.searchsorted(cumsum, positions)
        
    elif method == 'multinomial':
        # Multinomial resampling
        indices = np.random.choice(n_particles, size=n_samples, p=weights)
        
    elif method == 'stratified':
        # Stratified resampling
        positions = (np.arange(n_samples) + np.random.uniform(size=n_samples)) / n_samples
        cumsum = np.cumsum(weights)
        indices = np.searchsorted(cumsum, positions)
        
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    return particles[indices]


# ============================================================================
# POMDP-specific Utilities
# ============================================================================

def create_observation_model(observation_type: str, 
                        noise_params: Dict) -> Callable:
    """
    Create observation model for POMDP.
    
    Args:
        observation_type: Type of observation ('gaussian', 'range', 'bearing')
        noise_params: Parameters for observation noise
        
    Returns:
        Observation likelihood function
    """
    if observation_type == 'gaussian':
        noise_cov = noise_params['covariance']
        
        def gaussian_observation(state: np.ndarray, observation: np.ndarray) -> float:
            try:
                return multivariate_normal.pdf(observation, mean=state, cov=noise_cov)
            except Exception:
                return 1e-10
                
        return gaussian_observation
    
    elif observation_type == 'range':
        noise_std = noise_params['std']
        
        def range_observation(state: np.ndarray, observation: float) -> float:
            true_range = np.linalg.norm(state[:2])  # Assuming 2D position
            return np.exp(-0.5 * ((observation - true_range) / noise_std)**2) / (np.sqrt(2*np.pi) * noise_std)
            
        return range_observation
    
    elif observation_type == 'bearing':
        noise_std = noise_params['std']
        
        def bearing_observation(state: np.ndarray, observation: float) -> float:
            true_bearing = np.arctan2(state[1], state[0])
            # Handle angle wrapping
            diff = observation - true_bearing
            diff = (diff + np.pi) % (2*np.pi) - np.pi
            return np.exp(-0.5 * (diff / noise_std)**2) / (np.sqrt(2*np.pi) * noise_std)
            
        return bearing_observation
    
    else:
        raise ValueError(f"Unknown observation type: {observation_type}")


def create_transition_model(motion_type: str,
                        motion_params: Dict) -> Callable:
    """
    Create transition model for POMDP.
    
    Args:
        motion_type: Type of motion ('linear', 'unicycle', 'random_walk')
        motion_params: Parameters for motion model
        
    Returns:
        State transition function
    """
    if motion_type == 'linear':
        dt = motion_params.get('dt', 0.1)
        noise_std = motion_params.get('noise_std', 0.01)
        
        def linear_transition(state: np.ndarray, action: np.ndarray) -> np.ndarray:
            # Simple linear dynamics
            next_state = state + dt * action
            next_state += np.random.normal(0, noise_std, size=state.shape)
            return next_state
            
        return linear_transition
    
    elif motion_type == 'unicycle':
        dt = motion_params.get('dt', 0.1)
        noise_std = motion_params.get('noise_std', 0.01)
        
        def unicycle_transition(state: np.ndarray, action: np.ndarray) -> np.ndarray:
            # state = [x, y, theta], action = [v, omega]
            x, y, theta = state[:3]
            v, omega = action[:2]
            
            # Update state
            next_x = x + v * np.cos(theta) * dt
            next_y = y + v * np.sin(theta) * dt
            next_theta = theta + omega * dt
            
            next_state = np.array([next_x, next_y, next_theta])
            if len(state) > 3:
                next_state = np.concatenate([next_state, state[3:]])
            
            # Add noise
            next_state += np.random.normal(0, noise_std, size=state.shape)
            return next_state
            
        return unicycle_transition
    
    elif motion_type == 'random_walk':
        step_size = motion_params.get('step_size', 0.1)
        
        def random_walk_transition(state: np.ndarray, action: np.ndarray) -> np.ndarray:
            # Random walk with bias from action
            next_state = state + step_size * action
            next_state += np.random.normal(0, step_size, size=state.shape)
            return next_state
            
        return random_walk_transition
    
    else:
        raise ValueError(f"Unknown motion type: {motion_type}")
