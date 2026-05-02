import torch
import numpy as np
from typing import Dict, Optional, Any, Tuple, Callable
from dataclasses import dataclass


@dataclass
class ESCORTConfig:
    """Configuration for ESCORT agent."""
    # Particle settings
    n_particles: int = 100
    state_dim: int = 2
    particle_init_strategy: str = 'multi_modal'  # 'multi_modal', 'gaussian', 'uniform'
    
    # Kernel settings
    kernel_bandwidth: float = 1.0
    adaptive_kernel: bool = True
    kernel_bandwidth_scale: float = 0.5
    
    # SVGD settings
    svgd_step_size: float = 0.01
    svgd_max_iter: int = 50
    svgd_tolerance: float = 1e-5
    
    # Regularization weights
    lambda_corr: float = 0.1  # Correlation-aware regularization
    lambda_temp: float = 0.1  # Temporal consistency
    
    # GSWD settings
    n_projections: int = 10
    projection_method: str = 'optimized'  # 'random', 'optimized', 'pca'
    projection_optimization_steps: int = 5
    
    # Policy network settings
    policy_hidden_dim: int = 128
    policy_num_layers: int = 2
    policy_num_heads: int = 4
    policy_dropout: float = 0.0
    policy_lr: float = 3e-4
    discrete_actions: bool = True
    
    # Temporal consistency settings
    temporal_history_window: int = 5
    adaptive_temporal_lambda: bool = True
    max_temporal_distance: float = 1.0
    
    # General settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    verbose: bool = False
    random_seed: Optional[int] = None


class ESCORTAgent:
    """
    Main ESCORT agent implementing Efficient Stein-variational and 
    Sliced Consistency-Optimized Temporal Belief Representation for POMDPs.
    
    This class integrates all ESCORT components to provide a complete
    solution for belief tracking and decision-making in POMDPs.
    """
    
    def __init__(self, config: ESCORTConfig):
        """
        Initialize ESCORT agent with configuration.
        
        Args:
            config: ESCORTConfig object with all settings
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seed if provided
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config.random_seed)
        
        # Initialize components (imported from other modules)
        self._initialize_components()
        
        # State tracking
        self.initialized = False
        self.action_dim = None
        self.episode_count = 0
        self.total_steps = 0
        
        # Performance tracking
        self.belief_history = []
        self.action_history = []
        self.reward_history = []
        self.loss_history = []
    
    
    def _initialize_components(self):
        """Initialize all ESCORT components."""
        # Import components (assuming they're in the escort package)
        from .kernel import ImprovedRBFKernel
        from .gswd import ImprovedGSWD
        from .svgd import ImprovedSVGD
        from .particles import ParticleManager
        from .belief_updater import BeliefUpdater
        from .temporal import TemporalConsistency
        
        # Initialize kernel
        self.kernel = ImprovedRBFKernel(
            bandwidth=self.config.kernel_bandwidth,
            adaptive=self.config.adaptive_kernel,
            bandwidth_scale=self.config.kernel_bandwidth_scale
        )
        
        # Initialize GSWD
        self.gswd = ImprovedGSWD(
            n_projections=self.config.n_projections,
            projection_method=self.config.projection_method,
            optimization_steps=self.config.projection_optimization_steps,
            correlation_aware=True
        )
        
        # Initialize particle manager
        self.particle_manager = ParticleManager(
            n_particles=self.config.n_particles,
            state_dim=self.config.state_dim,
            initialization_strategy=self.config.particle_init_strategy,
            random_seed=self.config.random_seed
        )
        
        # Initialize SVGD
        self.svgd = ImprovedSVGD(
            kernel=self.kernel,
            gswd=self.gswd,
            step_size=self.config.svgd_step_size,
            lambda_corr=self.config.lambda_corr,
            lambda_temp=self.config.lambda_temp,
            max_iter=self.config.svgd_max_iter,
            tol=self.config.svgd_tolerance,
            verbose=self.config.verbose
        )
        
        # Initialize belief updater
        self.belief_updater = BeliefUpdater(
            svgd_updater=self.svgd,
            particle_manager=self.particle_manager,
            kernel=self.kernel,
            gswd=self.gswd,
            step_size=self.config.svgd_step_size,
            lambda_corr=self.config.lambda_corr,
            lambda_temp=self.config.lambda_temp,
            verbose=self.config.verbose
        )
        
        # Initialize temporal consistency
        self.temporal_consistency = TemporalConsistency(
            gswd=self.gswd,
            max_temporal_distance=self.config.max_temporal_distance,
            adaptive_lambda=self.config.adaptive_temporal_lambda,
            history_window=self.config.temporal_history_window,
            verbose=self.config.verbose
        )
        
        # Policy network will be initialized when action dimension is known
        self.policy_network = None
        self.policy_optimizer = None
    

    def _initialize_policy(self, action_dim: int):
        """
        Initialize policy network once action dimension is known.
        
        Args:
            action_dim: Dimension of action space
        """
        from .policy_network import PolicyNetwork, PolicyOptimizer
        
        self.action_dim = action_dim
        
        # Create policy network
        self.policy_network = PolicyNetwork(
            state_dim=self.config.state_dim,
            action_dim=action_dim,
            hidden_dim=self.config.policy_hidden_dim,
            num_encoder_layers=self.config.policy_num_layers,
            num_heads=self.config.policy_num_heads,
            discrete_actions=self.config.discrete_actions,
            dropout=self.config.policy_dropout
        ).to(self.device)
        
        # Create optimizer
        self.policy_optimizer = PolicyOptimizer(
            policy_network=self.policy_network,
            learning_rate=self.config.policy_lr,
            entropy_coef=0.01,
            max_grad_norm=0.5
        )
        
        self.initialized = True
        
        if self.config.verbose:
            print(f"Initialized ESCORT policy with action_dim={action_dim}")
    

    def update_belief(self,
            action: Any,
            observation: Any,
            transition_model: Callable,
            observation_model: Callable) -> np.ndarray:
        """Update belief based on action and observation."""
        
        # Update belief using belief updater
        particles = self.belief_updater.update(
            action, observation, transition_model, observation_model
        )
        
        # Update temporal consistency history
        self.temporal_consistency.update_history(
            particles, action, observation, self.total_steps
        )
        
        # Track belief
        self.belief_history.append(particles.copy())
        
        self.total_steps += 1
        
        return particles
    

    def select_action(self,
                    action_space: Optional[Any] = None,
                    deterministic: bool = False) -> Tuple[Any, Dict]:
        """
        Select action based on current belief.
        
        Args:
            action_space: Action space (needed for first call)
            deterministic: Whether to act deterministically
            
        Returns:
            Tuple of (action, info_dict)
        """
        # Initialize policy if needed
        if not self.initialized and action_space is not None:
            if self.config.discrete_actions:
                action_dim = action_space.n if hasattr(action_space, 'n') else len(action_space)
            else:
                action_dim = action_space.shape[0]
            self._initialize_policy(action_dim)
        
        if not self.initialized:
            raise ValueError("Policy not initialized. Provide action_space on first call.")
        
        # Get current particles
        particles = self.particle_manager.get_particles()
        
        # Convert to torch tensor
        particles_tensor = torch.FloatTensor(particles).unsqueeze(0).to(self.device)
        
        # Get action from policy
        action, info = self.policy_network.get_action(
            particles_tensor, deterministic=deterministic
        )
        
        # Convert to numpy if needed
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().squeeze()
        
        # Track action
        self.action_history.append(action)
        
        return action, info
    

    def train_step(self,
                particles: np.ndarray,
                actions: np.ndarray,
                rewards: np.ndarray,
                next_particles: np.ndarray,
                dones: np.ndarray) -> Dict:
        """
        Perform a training step for the policy.
        
        Args:
            particles: Batch of belief particles
            actions: Actions taken
            rewards: Rewards received
            next_particles: Next belief particles
            dones: Episode termination flags
            
        Returns:
            Training statistics
        """
        if not self.initialized:
            raise ValueError("Agent not initialized. Call select_action first.")
        
        # Convert to torch tensors
        particles = torch.FloatTensor(particles).to(self.device)
        actions = torch.LongTensor(actions).to(self.device) if self.config.discrete_actions else torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_particles = torch.FloatTensor(next_particles).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute advantages (simple version - can be replaced with GAE)
        with torch.no_grad():
            # Estimate values using average reward
            values = rewards  # Simplified - in practice, use value network
            next_values = torch.zeros_like(rewards)
            next_values[~dones.bool()] = rewards[~dones.bool()].mean()
            
            advantages = rewards + 0.99 * next_values - values
        
        # Policy gradient update
        info = self.policy_optimizer.update(
            particles, actions, advantages
        )
        
        # Track loss
        self.loss_history.append(info['total_loss'])
        
        return info
    

    def get_belief_statistics(self) -> Dict:
        """
        Get comprehensive statistics about current belief.
        
        Returns:
            Dictionary of belief statistics
        """
        # Get basic statistics from particle manager
        stats = self.particle_manager.compute_mode_statistics()
        
        # Add temporal statistics
        temporal_stats = self.temporal_consistency.get_statistics()
        stats.update({f'temporal_{k}': v for k, v in temporal_stats.items()})
        
        # Add kernel statistics
        if hasattr(self.kernel, 'adaptive_bandwidth') and self.kernel.adaptive_bandwidth is not None:
            stats['kernel_bandwidth'] = self.kernel.adaptive_bandwidth
        
        # Add policy statistics if initialized
        if self.initialized:
            stats['policy_initialized'] = True
            stats['total_steps'] = self.total_steps
            stats['episode_count'] = self.episode_count
        
        return stats
    

    def reset(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset agent for new episode.
        
        Args:
            initial_state: Optional initial state for particles
            
        Returns:
            Initial belief particles
        """
        # Reset particle manager
        if initial_state is not None:
            # Initialize particles around initial state
            initial_particles = np.random.normal(
                initial_state, 0.1, 
                (self.config.n_particles, self.config.state_dim)
            )
            self.particle_manager.reset(initial_particles)
        else:
            self.particle_manager.reset()
        
        # Reset belief updater
        self.belief_updater.reset()
        
        # Reset temporal consistency
        self.temporal_consistency.reset()
        
        # Clear episode history
        self.belief_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        
        self.episode_count += 1
        
        return self.particle_manager.get_particles()
    

    def save(self, path: str):
        """
        Save agent state to file.
        
        Args:
            path: Path to save file
        """
        state = {
            'config': self.config,
            'particles': self.particle_manager.get_particles(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count
        }
        
        if self.initialized:
            state['policy_state_dict'] = self.policy_network.state_dict()
            state['optimizer_state_dict'] = self.policy_optimizer.optimizer.state_dict()
        
        torch.save(state, path)
        
        if self.config.verbose:
            print(f"Saved ESCORT agent to {path}")
    

    def load(self, path: str):
        """
        Load agent state from file.
        
        Args:
            path: Path to save file
        """
        state = torch.load(path, map_location=self.device)
        
        # Restore config
        self.config = state['config']
        
        # Restore particles
        self.particle_manager.reset(state['particles'])
        
        # Restore counters
        self.total_steps = state['total_steps']
        self.episode_count = state['episode_count']
        
        # Restore policy if available
        if 'policy_state_dict' in state:
            # Initialize policy with correct action dimension
            action_dim = state['policy_state_dict']['action_head.weight'].shape[0] if self.config.discrete_actions else state['policy_state_dict']['action_mean.weight'].shape[0]
            self._initialize_policy(action_dim)
            
            self.policy_network.load_state_dict(state['policy_state_dict'])
            self.policy_optimizer.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        if self.config.verbose:
            print(f"Loaded ESCORT agent from {path}")
    

    def visualize_belief(self, save_path: Optional[str] = None):
        """
        Visualize current belief distribution.
        
        Args:
            save_path: Optional path to save visualization
        """
        from .utils import visualize_particle_belief
        
        particles = self.particle_manager.get_particles()
        mode_stats = self.particle_manager.compute_mode_statistics()
        
        visualize_particle_belief(
            particles,
            mode_stats,
            title=f"ESCORT Belief (Step {self.total_steps})",
            save_path=save_path
        )
