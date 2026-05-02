from typing import Dict, Optional
import warnings

import numpy as np
from sklearn.cluster import KMeans


class ParticleManager:
    """
        Manages particle initialization, storage, and statistics for ESCORT belief representation.
        
        This class handles all particle-related operations including initialization strategies
        for better multi-modal coverage, mode detection, and belief statistics computation.
    """
    def __init__(self, n_particles: int, state_dim: int, 
                initialization_strategy: str = 'multi_modal',
                random_seed: Optional[int] = None):
        """
            Initialize the particle manager.
            
            Args:
                n_particles: Number of particles for belief representation
                state_dim: Dimensionality of the state space
                initialization_strategy: Strategy for particle initialization 
                                    ('multi_modal', 'gaussian', 'uniform')
                random_seed: Random seed for reproducibility
        """
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.initialization_strategy = initialization_strategy
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # & Initialize particles
        self.particles = self.initialize_particles()
        
        # & Store previous particles for temporal consistency
        self.prev_particles = self.particles.copy()
        
        # & Cache for mode statistics
        self._mode_stats_cache = None
        self._cache_valid = False


    def initialize_particles(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
            Initialize particles with specified strategy for better coverage of the state space.
            
            Args:
                initial_state: Optional initial state to center particles around
                
            Returns:
                np.ndarray: Initial particle states of shape (n_particles, state_dim)
        """
        if self.initialization_strategy == 'multi_modal':
            particles = self._initialize_multi_modal()
        elif self.initialization_strategy == 'gaussian':
            particles = self._initialize_gaussian(initial_state)
        elif self.initialization_strategy == 'uniform':
            particles = self._initialize_uniform()
        else:
            raise ValueError(f"Unknown initialization strategy: {self.initialization_strategy}")
        
        # & Ensure particles are within reasonable bounds
        particles = self._clip_particles(particles)
        
        return particles
    

    def _initialize_multi_modal(self) -> np.ndarray:
        """
            Initialize particles with multi-modal coverage strategy.
            
            This strategy is particularly effective for low-dimensional spaces where
            we want to ensure coverage of different regions of the state space.
            
            Returns:
                np.ndarray: Initialized particles
        """
        particles = np.zeros((self.n_particles, self.state_dim))
        
        if self.state_dim <= 3:
            # & For low-dimensional cases, ensure better multi-modal coverage
            n_groups = min(4, 2**self.state_dim)
            particles_per_group = self.n_particles // n_groups
            remainder = self.n_particles - particles_per_group * n_groups
            
            # & Define group centers based on state space corners
            group_centers = []
            for i in range(n_groups):
                center = np.zeros(self.state_dim)
                # & Create binary pattern for corners
                for j in range(self.state_dim):
                    center[j] = 1.0 if (i & (1 << j)) > 0 else 0.0
                group_centers.append(center)
            
            # & Place particles around group centers
            for i in range(n_groups):
                start_idx = i * particles_per_group
                end_idx = start_idx + particles_per_group
                
                # & Add noise scaled by dimensionality
                noise_scale = 0.1 * (1.0 + 0.05 * self.state_dim)
                particles[start_idx:end_idx] = (
                    group_centers[i] + 
                    np.random.randn(particles_per_group, self.state_dim) * noise_scale
                )
            
            # & Randomly place remaining particles
            if remainder > 0:
                particles[-remainder:] = np.random.rand(remainder, self.state_dim)
        else:
            # & For high-dimensional cases, use a mixture of strategies
            half = self.n_particles // 2
            particles[:half] = np.random.normal(0.5, 0.1, (half, self.state_dim))
            
            # & Other half from wider distribution for exploration
            particles[half:] = np.random.normal(0.5, 0.3, (self.n_particles - half, self.state_dim))
        
        return particles
    

    def _initialize_gaussian(self, center: Optional[np.ndarray] = None) -> np.ndarray:
        """
            Initialize particles from a Gaussian distribution.
            
            Args:
                center: Center of the Gaussian distribution
                
            Returns:
                np.ndarray: Initialized particles
        """
        if center is None:
            center = np.full(self.state_dim, 0.5)
        
        # & Scale standard deviation with dimensionality
        std_dev = 0.1 * np.sqrt(1.0 + 0.1 * self.state_dim)
        
        particles = np.random.normal(
            center, 
            std_dev, 
            (self.n_particles, self.state_dim)
        )
        
        return particles
    

    def _initialize_uniform(self) -> np.ndarray:
        """
            Initialize particles uniformly in the unit hypercube.
            
            Returns:
                np.ndarray: Initialized particles
        """
        return np.random.uniform(0, 1, (self.n_particles, self.state_dim))


    def _clip_particles(self, particles: np.ndarray, 
                    min_val: float = -10.0, 
                    max_val: float = 10.0) -> np.ndarray:
        """
            Clip particle values to prevent numerical issues.
            
            Args:
                particles: Particle states
                min_val: Minimum allowed value
                max_val: Maximum allowed value
                
            Returns:
                np.ndarray: Clipped particles
        """
        return np.clip(particles, min_val, max_val)
    

    def update_particles(self, new_particles: np.ndarray) -> None:
        """
            Update particle states and invalidate cached statistics.
            
            Args:
                new_particles: New particle states
        """
        # & Validate input
        if new_particles.shape != (self.n_particles, self.state_dim):
            raise ValueError(
                f"Expected particles shape ({self.n_particles}, {self.state_dim}), "
                f"got {new_particles.shape}"
            )
        
        # & Check for numerical issues
        if not np.all(np.isfinite(new_particles)):
            warnings.warn("Non-finite values detected in particles. Applying fixes.")
            new_particles = np.nan_to_num(
                new_particles,
                nan=0.5,
                posinf=10.0,
                neginf=-10.0
            )
        
        # & Store previous particles
        self.prev_particles = self.particles.copy()
        
        # & Update particles
        self.particles = self._clip_particles(new_particles)
        
        # & Invalidate cache
        self._cache_valid = False

    
    def get_particles(self) -> np.ndarray:
        """
            Get current particle states.
            
            Returns:
                np.ndarray: Current particles
        """
        return self.particles.copy()
    

    def get_previous_particles(self) -> np.ndarray:
        """
            Get previous particle states.
            
            Returns:
                np.ndarray: Previous particles
        """
        return self.prev_particles.copy()


    def compute_mode_statistics(self, force_recompute: bool = False) -> Dict:
        """
            Compute statistics about the modes in the belief distribution.
            
            Uses improved clustering to identify distinct modes and their properties.
            
            Args:
                force_recompute: Whether to force recomputation even if cached
                
            Returns:
                dict: Statistics including number of modes, weights, centers, and covariances
        """
        # & Return cached results if valid
        if self._cache_valid and not force_recompute:
            return self._mode_stats_cache
        
        try:
            # & Estimate optimal number of clusters
            n_clusters = self._estimate_n_clusters()
            
            # & Perform clustering
            kmeans = KMeans(
                n_clusters=n_clusters, 
                n_init=10, 
                random_state=42
            )
            labels = kmeans.fit_predict(self.particles)
            
            # & Compute statistics for each cluster
            stats = self._compute_cluster_statistics(labels, kmeans.cluster_centers_)
            
            # & Cache results
            self._mode_stats_cache = stats
            self._cache_valid = True
            
            return stats
            
        except Exception as e:
            warnings.warn(f"Error in mode detection: {e}. Using fallback.")
            return self._compute_fallback_statistics()
        

    def _estimate_n_clusters(self) -> int:
        """
            Estimate the optimal number of clusters based on dimensionality and data.
            
            Returns:
                int: Estimated number of clusters
        """
        # & Base estimate on dimensionality
        n_clusters = min(8, max(2, self.state_dim + 1))
        
        # & Adjust based on particle spread if needed
        if self.n_particles < 50:
            n_clusters = min(n_clusters, self.n_particles // 10 + 1)
        
        return n_clusters


    def _compute_cluster_statistics(self, labels: np.ndarray, 
                                centers: np.ndarray) -> Dict:
        """
            Compute detailed statistics for each cluster.
            
            Args:
                labels: Cluster labels for each particle
                centers: Cluster centers
                
            Returns:
                dict: Cluster statistics
        """
        unique_labels = np.unique(labels)
        n_modes = len(unique_labels)
        
        weights = np.zeros(n_modes)
        covariances = []
        spreads = np.zeros(n_modes)
        
        for i, label in enumerate(unique_labels):
            # & Get particles in this cluster
            cluster_mask = labels == label
            cluster_particles = self.particles[cluster_mask]
            n_cluster = len(cluster_particles)
            
            # & Compute weight (proportion of particles)
            weights[i] = n_cluster / self.n_particles
            
            # & Compute covariance with regularization
            if n_cluster > 1:
                # & Center the data
                centered = cluster_particles - centers[i]
                
                # & Compute covariance
                cov = np.dot(centered.T, centered) / (n_cluster - 1)
                
                # & Add regularization for numerical stability
                reg = 1e-6 * np.eye(self.state_dim)
                cov = cov + reg
                
                # & Ensure symmetry
                cov = (cov + cov.T) / 2
                
                # & Compute spread (trace of covariance)
                spreads[i] = np.trace(cov)
            else:
                # & Single particle cluster
                cov = np.eye(self.state_dim) * 0.01
                spreads[i] = 0.01 * self.state_dim
            
            covariances.append(cov)
        
        return {
            'num_modes': n_modes,
            'mode_weights': weights,
            'mode_centers': centers,
            'mode_covariances': covariances,
            'mode_spreads': spreads,
            'labels': labels
        }


    def _compute_fallback_statistics(self) -> Dict:
        """
            Compute simple fallback statistics when clustering fails.
            
            Returns:
                dict: Basic statistics
        """
        mean = np.mean(self.particles, axis=0)
        cov = np.cov(self.particles, rowvar=False)
        
        # & Add regularization
        if self.state_dim > 1:
            cov = cov + 1e-6 * np.eye(self.state_dim)
        else:
            cov = np.array([[max(cov, 1e-6)]])
        
        return {
            'num_modes': 1,
            'mode_weights': np.array([1.0]),
            'mode_centers': np.array([mean]),
            'mode_covariances': [cov],
            'mode_spreads': np.array([np.trace(cov)]),
            'labels': np.zeros(self.n_particles, dtype=int)
        }


    def get_belief_entropy(self) -> float:
        """
            Estimate the entropy of the belief distribution.
            
            Returns:
                float: Estimated entropy
        """
        mode_stats = self.compute_mode_statistics()
        
        # & Compute entropy based on mode weights
        weights = mode_stats['mode_weights']
        
        # & Avoid log(0)
        weights = np.maximum(weights, 1e-10)
        
        # & Shannon entropy of mode weights
        mode_entropy = -np.sum(weights * np.log(weights))
        
        # & Add contribution from mode spreads
        spreads = mode_stats['mode_spreads']
        spread_entropy = np.sum(weights * np.log(spreads + 1))
        
        return mode_entropy + 0.5 * spread_entropy


    def resample_low_density_particles(self, densities: np.ndarray, 
                                    threshold: float = 0.1) -> None:
        """
            Resample particles in low-density regions to maintain diversity.
            
            Args:
                densities: Density estimates for each particle
                threshold: Threshold below which particles are considered low-density
        """
        # & Normalize densities
        max_density = np.max(densities)
        if max_density > 0:
            normalized_densities = densities / max_density
        else:
            return  # & All particles have zero density, skip resampling
        
        # & Find low and high density particles
        low_density_mask = normalized_densities < threshold
        high_density_mask = normalized_densities > np.median(normalized_densities)
        
        low_indices = np.where(low_density_mask)[0]
        high_indices = np.where(high_density_mask)[0]
        
        if len(low_indices) == 0 or len(high_indices) == 0:
            return  # & No resampling needed
        
        # & Resample low-density particles
        for idx in low_indices:
            # & Sample from high-density particles
            source_idx = np.random.choice(high_indices)
            
            # & Copy with noise to maintain diversity
            noise_scale = 0.1 * (1.0 + 0.05 * self.state_dim)
            noise = np.random.randn(self.state_dim) * noise_scale
            
            self.particles[idx] = self.particles[source_idx] + noise
        
        # & Clip particles after resampling
        self.particles = self._clip_particles(self.particles)
        
        # & Invalidate cache
        self._cache_valid = False


    def get_effective_sample_size(self) -> float:
        """
            Compute the effective sample size (ESS) of the particle set.
            
            Returns:
                float: Effective sample size
        """
        # & Compute mode statistics
        mode_stats = self.compute_mode_statistics()
        weights = mode_stats['mode_weights']
        
        # & ESS based on mode weights
        if len(weights) > 0:
            ess = 1.0 / np.sum(weights**2)
        else:
            ess = 1.0
        
        # & Scale by number of particles
        return ess * self.n_particles / len(weights) if len(weights) > 0 else self.n_particles


    def reset(self, initial_particles: Optional[np.ndarray] = None) -> None:
        """
            Reset the particle manager state.
            
            Args:
                initial_particles: Optional initial particle states
        """
        if initial_particles is not None:
            if initial_particles.shape != (self.n_particles, self.state_dim):
                raise ValueError(
                    f"Initial particles must have shape ({self.n_particles}, {self.state_dim})"
                )
            self.particles = initial_particles.copy()
        else:
            self.particles = self.initialize_particles()
        
        self.prev_particles = self.particles.copy()
        self._cache_valid = False
