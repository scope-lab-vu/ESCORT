
import warnings
import numpy as np
from typing import Optional, Dict, List
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class TemporalConsistency:
    """
    Temporal consistency regularization for ESCORT belief updates.
    
    This class implements temporal constraints that prevent unrealistic belief jumps
    between consecutive timesteps, as described in Section 3.4 of the ESCORT paper.
    It uses optimal transport principles to measure and regularize temporal changes.
    """
    
    def __init__(self,
                gswd=None,
                max_temporal_distance: float = 1.0,
                adaptive_lambda: bool = True,
                history_window: int = 5,
                verbose: bool = False):
        """
        Initialize temporal consistency module.
        
        Args:
            gswd: GSWD instance for projection-based distance computation
            max_temporal_distance: Maximum allowed belief change between timesteps
            adaptive_lambda: Whether to adapt regularization strength based on belief changes
            history_window: Number of past beliefs to consider for consistency
            verbose: Whether to display progress information
        """
        self.gswd = gswd
        self.max_temporal_distance = max_temporal_distance
        self.adaptive_lambda = adaptive_lambda
        self.history_window = history_window
        self.verbose = verbose
        
        # History tracking
        self.belief_history = []
        self.action_history = []
        self.observation_history = []
        self.timestamp_history = []
        
        # Adaptive parameters
        self.base_lambda = 0.1
        self.current_lambda = self.base_lambda
        self.lambda_history = []
        
        # Statistics tracking
        self.temporal_distances = []
        self.violation_count = 0
    
    def compute_temporal_regularization(self,
                                    current_particles: np.ndarray,
                                    previous_particles: np.ndarray,
                                    lambda_temp: Optional[float] = None) -> np.ndarray:
        """
        Compute temporal consistency regularization term.
        
        This implements Equation (6) from the paper:
        L_temp = ∫_Θ W_1((A_θ)^T b_{t+1}, (A_θ)^T b_t) dλ(θ)
        
        Args:
            current_particles: Current belief particles
            previous_particles: Previous timestep particles
            lambda_temp: Regularization strength (if None, uses adaptive)
            
        Returns:
            Regularization gradient for each particle
        """
        n_particles, dim = current_particles.shape
        
        # Use adaptive lambda if not provided
        if lambda_temp is None:
            lambda_temp = self._compute_adaptive_lambda(
                current_particles, previous_particles
            )
        
        # Method 1: GSWD-based regularization (if available)
        if self.gswd is not None:
            try:
                reg_term = self._gswd_temporal_regularization(
                    current_particles, previous_particles, lambda_temp
                )
            except Exception as e:
                warnings.warn(f"GSWD temporal regularization failed: {e}. Using fallback.")
                reg_term = self._ot_temporal_regularization(
                    current_particles, previous_particles, lambda_temp
                )
        else:
            # Method 2: Direct optimal transport regularization
            reg_term = self._ot_temporal_regularization(
                current_particles, previous_particles, lambda_temp
            )
        
        # Ensure numerical stability
        reg_term = self._ensure_finite_regularization(reg_term)
        
        # Track temporal distance
        temporal_dist = self._compute_temporal_distance(
            current_particles, previous_particles
        )
        self.temporal_distances.append(temporal_dist)
        
        # Check for violations
        if temporal_dist > self.max_temporal_distance:
            self.violation_count += 1
            if self.verbose:
                print(f"Temporal consistency violation: distance={temporal_dist:.4f}")
        
        return reg_term
    
    def _gswd_temporal_regularization(self,
                                    current: np.ndarray,
                                    previous: np.ndarray,
                                    lambda_temp: float) -> np.ndarray:
        """
        Compute temporal regularization using GSWD projections.
        
        Args:
            current: Current particles
            previous: Previous particles
            lambda_temp: Regularization strength
            
        Returns:
            Regularization term
        """
        # Fit GSWD if not already fitted
        if not self.gswd.fitted:
            self.gswd.fit(previous, current)
        
        # Get regularization term
        reg_term = self.gswd.get_regularizer(previous, current, lambda_temp)
        
        # Reverse direction (we want to regularize current toward previous)
        reg_term = -reg_term
        
        return reg_term
    
    def _ot_temporal_regularization(self,
                                current: np.ndarray,
                                previous: np.ndarray,
                                lambda_temp: float) -> np.ndarray:
        """
        Compute temporal regularization using optimal transport.
        
        Args:
            current: Current particles
            previous: Previous particles
            lambda_temp: Regularization strength
            
        Returns:
            Regularization term
        """
        n_particles = current.shape[0]
        dim = current.shape[1]
        
        # Compute pairwise distances between current and previous particles
        distances = cdist(current, previous, metric='euclidean')
        
        # Solve optimal transport problem (linear assignment)
        row_indices, col_indices = linear_sum_assignment(distances)
        
        # Compute regularization for each particle
        reg_term = np.zeros((n_particles, dim))
        
        for i, j in zip(row_indices, col_indices):
            # Direction from current to previous (to prevent large changes)
            direction = previous[j] - current[i]
            
            # Scale by distance and lambda
            distance = np.linalg.norm(direction)
            if distance > 0:
                # Normalize direction and scale
                direction_normalized = direction / distance
                
                # Stronger regularization for larger distances
                strength = lambda_temp * min(distance, self.max_temporal_distance)
                reg_term[i] = strength * direction_normalized
        
        return reg_term
    
    def _compute_adaptive_lambda(self,
                            current: np.ndarray,
                            previous: np.ndarray) -> float:
        """
        Compute adaptive regularization strength based on belief changes.
        
        Args:
            current: Current particles
            previous: Previous particles
            
        Returns:
            Adaptive lambda value
        """
        if not self.adaptive_lambda:
            return self.current_lambda
        
        # Compute temporal distance
        temporal_dist = self._compute_temporal_distance(current, previous)
        
        # Adapt lambda based on distance
        if temporal_dist > self.max_temporal_distance:
            # Increase regularization for large changes
            self.current_lambda = min(
                self.current_lambda * 1.5,
                self.base_lambda * 10
            )
        elif temporal_dist < self.max_temporal_distance * 0.5:
            # Decrease regularization for small changes
            self.current_lambda = max(
                self.current_lambda * 0.9,
                self.base_lambda * 0.1
            )
        
        # Track lambda history
        self.lambda_history.append(self.current_lambda)
        
        return self.current_lambda
    
    def _compute_temporal_distance(self,
                                current: np.ndarray,
                                previous: np.ndarray) -> float:
        """
        Compute Wasserstein distance between consecutive beliefs.
        
        Args:
            current: Current particles
            previous: Previous particles
            
        Returns:
            Temporal distance measure
        """
        # Method 1: Use GSWD if available
        if self.gswd is not None and self.gswd.fitted:
            try:
                distance = self.gswd.compute_distance(previous, current)
                return distance
            except Exception:
                pass
        
        # Method 2: Approximate using optimal transport
        distances = cdist(current, previous, metric='euclidean')
        row_indices, col_indices = linear_sum_assignment(distances)
        
        # Average distance of optimal assignment
        total_distance = distances[row_indices, col_indices].sum()
        avg_distance = total_distance / len(row_indices)
        
        return avg_distance
    
    def update_history(self,
                    particles: np.ndarray,
                    action: Optional[any] = None,
                    observation: Optional[any] = None,
                    timestamp: Optional[float] = None) -> None:
        """
        Update belief history for tracking temporal patterns.
        
        Args:
            particles: Current belief particles
            action: Action taken (optional)
            observation: Observation received (optional)
            timestamp: Current timestamp (optional)
        """
        # Add to history
        self.belief_history.append(particles.copy())
        self.action_history.append(action)
        self.observation_history.append(observation)
        self.timestamp_history.append(timestamp)
        
        # Maintain window size
        if len(self.belief_history) > self.history_window:
            self.belief_history.pop(0)
            self.action_history.pop(0)
            self.observation_history.pop(0)
            self.timestamp_history.pop(0)
    
    def compute_multi_step_consistency(self,
                                    current_particles: np.ndarray,
                                    horizon: int = 3) -> np.ndarray:
        """
        Compute consistency regularization over multiple past timesteps.
        
        Args:
            current_particles: Current belief particles
            horizon: Number of past steps to consider
            
        Returns:
            Multi-step regularization term
        """
        if len(self.belief_history) == 0:
            return np.zeros_like(current_particles)
        
        # Limit horizon to available history
        effective_horizon = min(horizon, len(self.belief_history))
        
        # Aggregate regularization over past steps
        total_reg = np.zeros_like(current_particles)
        total_weight = 0.0
        
        for i in range(effective_horizon):
            # Get historical belief (from most recent)
            hist_idx = -(i + 1)
            historical_belief = self.belief_history[hist_idx]
            
            # Compute weight (decay with time)
            weight = 1.0 / (i + 1)
            
            # Compute regularization
            reg = self._ot_temporal_regularization(
                current_particles,
                historical_belief,
                self.current_lambda * weight
            )
            
            total_reg += reg
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            total_reg /= total_weight
        
        return total_reg
    
    def get_temporal_smoothness_score(self) -> float:
        """
        Compute a smoothness score for the belief trajectory.
        
        Returns:
            Smoothness score (0-1, higher is smoother)
        """
        if len(self.temporal_distances) < 2:
            return 1.0
        
        # Compute variance of temporal distances
        distances = np.array(self.temporal_distances)
        
        # Normalize by maximum allowed distance
        normalized_distances = distances / self.max_temporal_distance
        
        # Compute smoothness (inverse of variance)
        variance = np.var(normalized_distances)
        smoothness = 1.0 / (1.0 + variance)
        
        return smoothness
    
    def detect_belief_jumps(self,
                        threshold_multiplier: float = 2.0) -> List[int]:
        """
        Detect timesteps with significant belief jumps.
        
        Args:
            threshold_multiplier: Multiplier for jump detection threshold
            
        Returns:
            List of timestep indices with detected jumps
        """
        if len(self.temporal_distances) < 2:
            return []
        
        distances = np.array(self.temporal_distances)
        threshold = self.max_temporal_distance * threshold_multiplier
        
        # Find jumps
        jump_indices = np.where(distances > threshold)[0]
        
        return jump_indices.tolist()
    
    def _ensure_finite_regularization(self,
                                    reg_term: np.ndarray) -> np.ndarray:
        """
        Ensure regularization term contains only finite values.
        
        Args:
            reg_term: Regularization term
            
        Returns:
            Cleaned regularization term
        """
        if not np.all(np.isfinite(reg_term)):
            warnings.warn("Non-finite values in temporal regularization. Fixing.")
            reg_term = np.nan_to_num(
                reg_term,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )
            
            # Clip to reasonable range
            max_reg = 10.0
            reg_term = np.clip(reg_term, -max_reg, max_reg)
        
        return reg_term
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get temporal consistency statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'current_lambda': self.current_lambda,
            'violation_count': self.violation_count,
            'smoothness_score': self.get_temporal_smoothness_score(),
            'history_length': len(self.belief_history)
        }
        
        if len(self.temporal_distances) > 0:
            distances = np.array(self.temporal_distances)
            stats.update({
                'mean_temporal_distance': np.mean(distances),
                'max_temporal_distance': np.max(distances),
                'std_temporal_distance': np.std(distances)
            })
        
        # Detect jumps
        jumps = self.detect_belief_jumps()
        stats['num_jumps'] = len(jumps)
        stats['jump_timesteps'] = jumps
        
        return stats
    
    def reset(self) -> None:
        """Reset temporal consistency state."""
        self.belief_history.clear()
        self.action_history.clear()
        self.observation_history.clear()
        self.timestamp_history.clear()
        self.temporal_distances.clear()
        self.lambda_history.clear()
        self.current_lambda = self.base_lambda
        self.violation_count = 0
        
        if self.verbose:
            print("Temporal consistency module reset.")
