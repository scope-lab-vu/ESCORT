from typing import Callable, Optional
import warnings
import numpy as np

class ImprovedSVGD:
    """Improved SVGD implementation with better multi-modal handling and numerical stability."""
    
    def __init__(self, kernel, gswd, step_size: float = 0.01,
                lambda_corr: float = 0.1, lambda_temp: float = 0.1,
                max_iter: int = 50, tol: float = 1e-5, verbose: bool = False):
        self.kernel = kernel
        self.gswd = gswd
        self.step_size = step_size
        self.lambda_corr = lambda_corr
        self.lambda_temp = lambda_temp
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def _compute_svgd_update(self, particles: np.ndarray, score_fn: Callable) -> np.ndarray:
        """Compute SVGD update with numerical safeguards - FULLY VECTORIZED."""
        n_particles, dim = particles.shape
        
        # Compute kernel matrix and gradient efficiently
        K = self.kernel.evaluate(particles)
        grad_K = self.kernel.gradient(particles)
        
        # Compute score function values
        try:
            score_values, _ = score_fn(particles, return_logp=True)
        except Exception:
            score_values = score_fn(particles)
        
        score_values = np.nan_to_num(score_values, nan=0.0, posinf=0.0, neginf=0.0)
        score_values = np.clip(score_values, -100.0, 100.0)
        
        # Vectorized update computation
        # Attractive term: sum over j of K(x_j, x_i) * score(x_j)
        attractive = np.einsum('ji,jd->id', K, score_values)
        
        # Repulsive term: sum over j of grad_K[j, i, :]
        repulsive = np.sum(grad_K, axis=0)
        
        # Enhance repulsive forces in high dimensions
        repulsion_factor = 1.0 + 0.1 * dim
        repulsive *= repulsion_factor
        
        # Combine and normalize
        update = (attractive + repulsive) / n_particles
        
        return np.clip(update, -10.0, 10.0)
    
    def _compute_temporal_consistency(self, particles: np.ndarray, 
                                    prev_particles: np.ndarray,
                                    lambda_temp: float) -> np.ndarray:
        """Compute temporal consistency regularization - OPTIMIZED."""
        if prev_particles is None or lambda_temp <= 0:
            return np.zeros_like(particles)
        
        try:
            # Use GSWD for efficient optimal transport-based regularization
            reg_term = self.gswd.get_regularizer(prev_particles, particles, lambda_temp)
            return np.nan_to_num(reg_term, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            warnings.warn(f"Error in temporal consistency: {e}. Using fallback.")
            # Vectorized fallback using broadcasting
            diff = prev_particles[:, np.newaxis, :] - particles[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            closest_indices = np.argmin(sq_dists, axis=0)
            reg_term = lambda_temp * (prev_particles[closest_indices] - particles)
            return reg_term
    
    def _detect_and_resample_particles(self, particles: np.ndarray, 
                                    score_fn: Callable) -> np.ndarray:
        """Detect and resample particles in low-density regions - OPTIMIZED."""
        n_particles, dim = particles.shape
        
        if n_particles < 10:
            return particles
        
        try:
            # Get density information
            _, log_probs = score_fn(particles, return_logp=True)
            densities = np.exp(np.clip(log_probs, -30, 30))
            
            # Normalize densities
            max_density = np.max(densities)
            if max_density > 0:
                normalized_densities = densities / max_density
            else:
                normalized_densities = np.ones(n_particles)
            
            # Find low-density particles
            threshold = 0.1
            low_density_mask = normalized_densities < threshold
            n_low = np.sum(low_density_mask)
            
            if n_low > 0:
                # Find high-density particles
                high_density_mask = normalized_densities > np.median(normalized_densities)
                high_indices = np.where(high_density_mask)[0]
                
                if len(high_indices) == 0:
                    high_indices = np.random.choice(n_particles, size=n_particles//2, replace=False)
                
                # Vectorized resampling
                low_indices = np.where(low_density_mask)[0]
                source_indices = np.random.choice(high_indices, size=len(low_indices))
                
                noise_scale = 0.1 * (1.0 + 0.05 * dim)
                noise = np.random.randn(len(low_indices), dim) * noise_scale
                
                particles[low_indices] = particles[source_indices] + noise
            
            return particles
            
        except Exception as e:
            warnings.warn(f"Error in particle resampling: {e}")
            return particles

    def update(self, particles: np.ndarray, score_fn: Callable,
            prev_particles: Optional[np.ndarray] = None,
            lambda_temp: Optional[float] = None) -> np.ndarray:
        """Update particles using improved SVGD - OPTIMIZED MAIN LOOP."""
        particles = particles.copy()
        lambda_temp = lambda_temp if lambda_temp is not None else self.lambda_temp
        
        # Pre-allocate for temporal update if needed
        if prev_particles is not None and lambda_temp > 0:
            temp_update = self._compute_temporal_consistency(
                particles, prev_particles, lambda_temp
            )
        
        for t in range(self.max_iter):
            old_particles = particles.copy()
            
            # Compute SVGD update
            svgd_update = self._compute_svgd_update(particles, score_fn)
            
            # Add temporal consistency if applicable
            if prev_particles is not None and lambda_temp > 0:
                update = svgd_update + temp_update
            else:
                update = svgd_update
            
            # Apply update
            particles = particles + self.step_size * update
            
            # Periodically resample (less frequently)
            if t > 0 and t % 20 == 0:
                particles = self._detect_and_resample_particles(particles, score_fn)
            
            # Check convergence
            diff = np.linalg.norm(particles - old_particles) / particles.shape[0]
            
            if diff < self.tol:
                if self.verbose:
                    print(f"SVGD converged after {t+1} iterations")
                break
        
        return particles
