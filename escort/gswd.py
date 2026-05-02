import warnings
import numpy as np
from scipy.linalg import eigh

class ImprovedGSWD:
    """Improved Generalized Sliced Wasserstein Distance with optimized projections."""
    
    def __init__(self, n_projections=10, projection_method='optimized', 
                optimization_steps=5, correlation_aware=True,
                learning_rate=0.01, momentum=0.9):
        self.n_projections = n_projections
        self.projection_method = projection_method
        self.optimization_steps = optimization_steps
        self.correlation_aware = correlation_aware
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.projections = None
        self.projection_weights = None
        self.covariance = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.fitted = False

    def _init_projections(self, dim):
        """Initialize projection directions."""
        if self.projection_method == 'random':
            projections = np.random.randn(self.n_projections, dim)
            # Vectorized normalization
            norms = np.linalg.norm(projections, axis=1, keepdims=True)
            projections = projections / (norms + 1e-10)
        elif self.projection_method == 'pca':
            projections = np.eye(dim)[:self.n_projections]
            if self.n_projections > dim:
                extra = np.random.randn(self.n_projections - dim, dim)
                extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
                projections = np.vstack([projections, extra])
        else:  # 'optimized'
            projections = np.random.randn(self.n_projections, dim)
            norms = np.linalg.norm(projections, axis=1, keepdims=True)
            projections = projections / (norms + 1e-10)
        
        weights = np.ones(self.n_projections) / self.n_projections
        return projections, weights
    
    def _estimate_covariance(self, samples):
        """Estimate covariance matrix from samples with robust regularization."""
        samples = np.nan_to_num(samples, nan=0.0, posinf=1e10, neginf=-1e10)
        centered = samples - np.mean(samples, axis=0, keepdims=True)
        n_samples = samples.shape[0]
        cov = np.dot(centered.T, centered) / (n_samples - 1)
        
        # Add regularization
        dim = samples.shape[1]
        cov = cov + 1e-6 * np.eye(dim)
        cov = (cov + cov.T) / 2  # Ensure symmetry
        
        return cov
    
    def _optimize_projections(self, source, target):
        """Optimize projection directions - VECTORIZED VERSION."""
        dim = source.shape[1]
        
        if self.projections is None or self.projections.shape[1] != dim:
            self.projections, self.projection_weights = self._init_projections(dim)
        
        projections = self.projections.copy()
        weights = self.projection_weights.copy()
        velocity = np.zeros_like(projections)
        
        for step in range(self.optimization_steps):
            # Vectorized projection and distance computation
            source_proj = source @ projections.T  # (n_source, n_projections)
            target_proj = target @ projections.T  # (n_target, n_projections)
            
            # Sort projections
            source_proj_sorted = np.sort(source_proj, axis=0)
            target_proj_sorted = np.sort(target_proj, axis=0)
            
            # Compute distances (1D Wasserstein)
            distances = np.mean(np.abs(source_proj_sorted - target_proj_sorted), axis=0)
            
            # Update weights
            weights = np.maximum(1e-3, distances)
            weights = weights / np.sum(weights)
            
            # Compute gradients using finite differences (vectorized)
            eps = 1e-6
            gradients = np.zeros_like(projections)
            
            for i in range(self.n_projections):
                # Vectorized gradient computation
                for j in range(dim):
                    proj_perturbed = projections[i].copy()
                    proj_perturbed[j] += eps
                    proj_perturbed = proj_perturbed / np.linalg.norm(proj_perturbed)
                    
                    # Compute perturbed projections
                    source_proj_pert = source @ proj_perturbed
                    target_proj_pert = target @ proj_perturbed
                    
                    # Sort and compute distance
                    source_sorted_pert = np.sort(source_proj_pert)
                    target_sorted_pert = np.sort(target_proj_pert)
                    dist_pert = np.mean(np.abs(source_sorted_pert - target_sorted_pert))
                    
                    gradients[i, j] = (dist_pert - distances[i]) / eps
            
            # Apply correlation-aware modification if enabled
            if self.correlation_aware and self.eigenvectors is not None:
                # Vectorized projection onto eigenvectors
                grad_proj = gradients @ self.eigenvectors
                grad_proj = grad_proj * np.sqrt(self.eigenvalues)
                gradients = grad_proj @ self.eigenvectors.T
            
            # Update with momentum
            learning_rate = self.learning_rate * (1.0 / (1.0 + 0.1 * step))
            velocity = self.momentum * velocity + learning_rate * gradients
            projections = projections + velocity
            
            # Normalize projections
            norms = np.linalg.norm(projections, axis=1, keepdims=True)
            projections = projections / (norms + 1e-10)
        
        return projections, weights
    
    def fit(self, source, target):
        """Fit GSWD to two point sets, optimizing projections."""
        source = np.nan_to_num(source, nan=0.0, posinf=1e10, neginf=-1e10)
        target = np.nan_to_num(target, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if self.correlation_aware:
            combined = np.vstack([source, target])
            self.covariance = self._estimate_covariance(combined)
            
            try:
                self.eigenvalues, self.eigenvectors = eigh(self.covariance)
                self.eigenvalues = np.maximum(self.eigenvalues, 1e-10)
                
                # Sort by descending eigenvalues
                idx = np.argsort(self.eigenvalues)[::-1]
                self.eigenvalues = self.eigenvalues[idx]
                self.eigenvectors = self.eigenvectors[:, idx]
                
                if self.projection_method == 'pca':
                    dim = source.shape[1]
                    self.projections = self.eigenvectors[:, :self.n_projections].T
                    
                    if self.projections.shape[0] < self.n_projections:
                        extra = np.random.randn(self.n_projections - self.projections.shape[0], dim)
                        # Orthogonalize using QR decomposition
                        Q, _ = np.linalg.qr(np.vstack([self.projections.T, extra.T]).T)
                        self.projections = Q[:, :self.n_projections].T
                    
                    self.projection_weights = self.eigenvalues[:self.n_projections]
                    self.projection_weights = self.projection_weights / np.sum(self.projection_weights)
            except np.linalg.LinAlgError:
                warnings.warn("Eigendecomposition failed. Using random projections.")
                self.projections, self.projection_weights = self._init_projections(source.shape[1])
        
        if self.projection_method != 'pca' or self.optimization_steps > 0:
            self.projections, self.projection_weights = self._optimize_projections(source, target)
        
        self.fitted = True

    def compute_distance(self, source, target, return_per_projection=False):
        """Compute the GSWD between two point sets - VECTORIZED."""
        source = np.nan_to_num(source, nan=0.0, posinf=1e10, neginf=-1e10)
        target = np.nan_to_num(target, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if not self.fitted or self.projections is None or self.projections.shape[1] != source.shape[1]:
            self.fit(source, target)
        
        # Vectorized projection
        source_proj = source @ self.projections.T  # (n_source, n_projections)
        target_proj = target @ self.projections.T  # (n_target, n_projections)
        
        # Sort along sample dimension
        source_proj_sorted = np.sort(source_proj, axis=0)
        target_proj_sorted = np.sort(target_proj, axis=0)
        
        # Compute 1D Wasserstein distances
        per_proj_distances = np.mean(np.abs(source_proj_sorted - target_proj_sorted), axis=0)
        
        # Weighted average
        total_distance = np.sum(per_proj_distances * self.projection_weights)
        
        if return_per_projection:
            return total_distance, per_proj_distances
        else:
            return total_distance

    def get_regularizer(self, source, target, lambda_reg=0.1):
        """Compute GSWD regularization term - FULLY VECTORIZED."""
        source = np.nan_to_num(source, nan=0.0, posinf=1e10, neginf=-1e10)
        target = np.nan_to_num(target, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if not self.fitted or self.projections is None or self.projections.shape[1] != source.shape[1]:
            self.fit(source, target)
        
        n_particles = target.shape[0]
        n_source = source.shape[0]
        dim = target.shape[1]
        
        # Project all points at once
        source_proj = source @ self.projections.T  # (n_source, n_projections)
        target_proj = target @ self.projections.T  # (n_target, n_projections)
        
        # Sort indices for each projection
        source_indices = np.argsort(source_proj, axis=0)  # (n_source, n_projections)
        target_indices = np.argsort(target_proj, axis=0)  # (n_target, n_projections)
        
        # Initialize regularization term
        reg_term = np.zeros((n_particles, dim))
        
        # Compute regularization for all projections at once
        for i in range(self.n_projections):
            weight = self.projection_weights[i]
            
            # Create optimal transport mapping
            source_idx = source_indices[:, i]
            target_idx = target_indices[:, i]
            
            # Map targets to sources (cycling if needed)
            mapped_source_idx = source_idx[np.arange(n_particles) % n_source]
            
            # Compute direction vectors
            directions = source[mapped_source_idx] - target[target_idx]
            
            # Add weighted contribution to regularization
            reg_term[target_idx] += lambda_reg * weight * directions
        
        return np.nan_to_num(reg_term, nan=0.0, posinf=0.0, neginf=0.0)
