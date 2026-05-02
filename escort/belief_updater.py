
import warnings
import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any


class BeliefUpdater:
    """
    ESCORT Belief Updater with correlation-aware regularization.
    
    This class implements the core belief update mechanism from Equation (2) of the paper:
    x^{t+1}_i = x^t_i + ε φ*_reg(x^t_i) + Update(o_{t+1}, a_t)
    
    It combines deterministic particle evolution with model-based state estimation
    to maintain accurate belief representations in POMDPs.
    """
    
    def __init__(self, 
                svgd_updater,
                particle_manager,
                kernel,
                gswd,
                step_size: float = 0.01,
                lambda_corr: float = 0.1,
                lambda_temp: float = 0.1,
                noise_scale_factor: float = 0.01,
                verbose: bool = False):
        """
        Initialize the belief updater.
        
        Args:
            svgd_updater: SVGD update mechanism
            particle_manager: Particle manager for belief representation
            kernel: Kernel function for SVGD
            gswd: GSWD instance for correlation-aware regularization
            step_size: Step size for belief updates
            lambda_corr: Weight for correlation-aware regularization
            lambda_temp: Weight for temporal consistency
            noise_scale_factor: Factor for transition noise scaling
            verbose: Whether to display progress information
        """
        self.svgd = svgd_updater
        self.particle_manager = particle_manager
        self.kernel = kernel
        self.gswd = gswd
        self.step_size = step_size
        self.lambda_corr = lambda_corr
        self.lambda_temp = lambda_temp
        self.noise_scale_factor = noise_scale_factor
        self.verbose = verbose
        
        # Cache for computational efficiency
        self._score_fn_cache = {}
        self._transition_cache = {}
    
    
    def update(self,
        action: Any,
        observation: Any,
        transition_model: Callable,
        observation_model: Callable) -> np.ndarray:
        """Update belief using the ESCORT framework."""
        
        # Get current particles from manager
        current_particles = self.particle_manager.get_particles()
        prev_particles = self.particle_manager.get_previous_particles()
        
        # Step 1: Apply transition model
        predicted_particles = self._apply_transition_model(
            current_particles, action, transition_model
        )
        
        # Step 2: Create score function
        score_fn = self._create_score_function(
            predicted_particles, observation, observation_model
        )
        
        # Step 3: Apply SVGD update
        updated_particles = self._apply_svgd_update(
            predicted_particles, score_fn, prev_particles
        )
        
        # Step 4: Update particle manager
        self.particle_manager.update_particles(updated_particles)
        
        return updated_particles
    

    def _apply_transition_model(self,
                            particles: np.ndarray,
                            action: Any,
                            transition_model: Callable) -> np.ndarray:
        """
        Apply transition model to particles with noise.
        
        Args:
            particles: Current particle states
            action: Action taken
            transition_model: State transition function
            
        Returns:
            Predicted particle states after transition
        """
        n_particles, state_dim = particles.shape
        predicted_particles = np.zeros_like(particles)
        
        try:
            # Apply transition for each particle
            for i in range(n_particles):
                # Clean particle from NaN or Inf
                clean_particle = np.nan_to_num(
                    particles[i], nan=0.0, posinf=1e10, neginf=-1e10
                )
                
                # Apply transition
                predicted_particles[i] = transition_model(clean_particle, action)
            
            # Add transition noise to prevent particle collapse
            # Scale noise with dimensionality
            noise_scale = self.noise_scale_factor * (1.0 + 0.1 * state_dim)
            transition_noise = np.random.randn(*predicted_particles.shape) * noise_scale
            predicted_particles += transition_noise
            
            # Handle any numerical issues
            predicted_particles = self._ensure_finite(
                predicted_particles, "predicted particles", fallback=particles
            )
            
            return predicted_particles
            
        except Exception as e:
            warnings.warn(f"Error in transition model: {e}. Using fallback.")
            # In case of error, add small noise to current particles
            noise_scale = self.noise_scale_factor * 0.5
            return particles + np.random.randn(*particles.shape) * noise_scale
    

    
    def _create_score_function(self,
                            particles: np.ndarray,
                            observation: Any,
                            observation_model: Callable) -> Callable:
        """Create score function with vectorized observation model calls."""
        state_dim = particles.shape[1]
        
        def score_fn(x: np.ndarray, return_logp: bool = False) -> Tuple:
            """Vectorized score function."""
            n_particles = x.shape[0]
            
            # Single vectorized call instead of 100 separate calls
            likelihoods = observation_model(x, observation)
            if np.isscalar(likelihoods):
                likelihoods = np.array([likelihoods])
            likelihoods = np.maximum(likelihoods, 1e-15)
            
            if return_logp:
                log_probs = np.log(likelihoods)
            
            # For high dimensions, use approximation
            scores = np.zeros((n_particles, state_dim))
            if state_dim > 50:
                weights = likelihoods / (likelihoods.sum() + 1e-10)
                center = np.sum(x * weights[:, np.newaxis], axis=0)
                
                for i in range(n_particles):
                    direction = center - x[i]
                    scores[i] = direction * likelihoods[i] * 0.1
            else:
                # Finite differences only for low dimensions
                eps = 1e-4
                for i in range(n_particles):
                    for d in range(state_dim):
                        x_pert = x[i].copy()
                        x_pert[d] += eps
                        lik_pert = observation_model(x_pert.reshape(1, -1), observation)[0]
                        lik_pert = max(lik_pert, 1e-15)
                        scores[i, d] = (np.log(lik_pert) - np.log(likelihoods[i])) / eps
            
            scores = np.clip(scores, -10.0, 10.0)
            
            if return_logp:
                return scores, log_probs
            else:
                return scores
        
        return score_fn
    

    def _compute_score_gradient(self,
                            particle: np.ndarray,
                            observation: Any,
                            observation_model: Callable,
                            base_likelihood: float) -> np.ndarray:
        """
        Compute gradient of log-likelihood using finite differences.
        
        Args:
            particle: Single particle state
            observation: Current observation
            observation_model: Observation likelihood function
            base_likelihood: Likelihood at current particle position
            
        Returns:
            Gradient vector
        """
        state_dim = len(particle)
        gradient = np.zeros(state_dim)
        
        # Adaptive step size based on particle scale
        scale = np.median(np.abs(particle)) if np.any(particle != 0) else 1.0
        eps = max(1e-6, 1e-4 * scale)
        
        # Second-order central difference for each dimension
        for d in range(state_dim):
            # Create perturbed particles
            particle_plus = particle.copy()
            particle_plus[d] += eps
            
            particle_minus = particle.copy()
            particle_minus[d] -= eps
            
            # Compute perturbed likelihoods
            likelihood_plus = observation_model(particle_plus, observation)
            likelihood_minus = observation_model(particle_minus, observation)
            
            # Ensure numerical stability
            likelihood_plus = max(likelihood_plus, 1e-15)
            likelihood_minus = max(likelihood_minus, 1e-15)
            
            # Central difference approximation
            gradient[d] = (np.log(likelihood_plus) - np.log(likelihood_minus)) / (2 * eps)
        
        return gradient
    

    
    def _apply_svgd_update(self,
                    particles: np.ndarray,
                    score_fn: Callable,
                    prev_particles: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply SVGD update with correlation-aware regularization."""
        # Configure SVGD with current regularization weights
        self.svgd.lambda_corr = self.lambda_corr
        self.svgd.lambda_temp = self.lambda_temp
        
        # Apply SVGD update
        if prev_particles is not None and self.lambda_temp > 0:
            # Update with temporal consistency
            updated_particles = self.svgd.update(
                particles,
                score_fn,
                prev_particles,
                self.lambda_temp
            )
        else:
            # Update without temporal consistency
            updated_particles = self.svgd.update(
                particles,
                score_fn
            )
        
        # Ensure particles remain in valid range
        updated_particles = self._ensure_finite(
            updated_particles, "updated particles", fallback=particles
        )
        
        return updated_particles
    

    def _ensure_finite(self,
                    array: np.ndarray,
                    name: str,
                    fallback: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Ensure array contains only finite values.
        
        Args:
            array: Array to check
            name: Name for error messages
            fallback: Fallback array if fixing fails
            
        Returns:
            Array with finite values
        """
        if not np.all(np.isfinite(array)):
            warnings.warn(f"NaN or Inf detected in {name}. Applying fix.")
            
            # Try to fix by replacing problematic values
            fixed_array = np.nan_to_num(
                array,
                nan=0.5,  # Replace NaN with neutral value
                posinf=10.0,
                neginf=-10.0
            )
            
            # If still has issues and fallback provided, use fallback
            if not np.all(np.isfinite(fixed_array)) and fallback is not None:
                # Replace only problematic entries from fallback
                bad_mask = ~np.all(np.isfinite(fixed_array), axis=-1)
                fixed_array[bad_mask] = fallback[bad_mask]
            
            return fixed_array
        
        return array
    

    def compute_correlation_regularization(self,
                                        particles: np.ndarray,
                                        target_distribution: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute correlation-aware regularization term.
        
        This implements Equation (3) from the paper:
        R_corr(φ) = E_x~q[Σ_i w_i · |A^T_i(φ(x) - E_y~q[φ(y)])|^2]
        
        Args:
            particles: Current particle distribution
            target_distribution: Target distribution (if None, uses uniform)
            
        Returns:
            Regularization gradients for each particle
        """
        if target_distribution is None:
            # Use a default target (e.g., uniform distribution)
            target_distribution = np.random.uniform(
                low=particles.min(axis=0),
                high=particles.max(axis=0),
                size=particles.shape
            )
        
        # Fit GSWD to learn optimal projections
        self.gswd.fit(particles, target_distribution)
        
        # Compute regularization term
        reg_term = self.gswd.get_regularizer(
            particles,
            target_distribution,
            self.lambda_corr
        )
        
        return reg_term
    

    def get_belief_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about current belief state.
        
        Returns:
            Dictionary containing belief statistics
        """
        stats = self.particle_manager.compute_mode_statistics()
        
        # Add entropy estimate
        stats['entropy'] = self.particle_manager.get_belief_entropy()
        
        # Add effective sample size
        stats['ess'] = self.particle_manager.get_effective_sample_size()
        
        return stats
    

    def reset(self, initial_particles: Optional[np.ndarray] = None) -> None:
        """
        Reset belief updater state.
        
        Args:
            initial_particles: Optional initial particle states
        """
        self.particle_manager.reset(initial_particles)
        self._score_fn_cache.clear()
        self._transition_cache.clear()
        
        if self.verbose:
            print("Belief updater reset.")
