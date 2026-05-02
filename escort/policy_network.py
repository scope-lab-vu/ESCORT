import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union


class ParticleEncoder(nn.Module):
    """
    Encoder network for processing individual particles.
    
    This module processes each particle independently to extract
    relevant features for decision-making.
    """
    
    def __init__(self, 
                state_dim: int,
                hidden_dim: int,
                num_layers: int = 2,
                dropout: float = 0.0):
        """
        Initialize particle encoder.
        
        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            num_layers: Number of encoding layers
            dropout: Dropout probability
        """
        super(ParticleEncoder, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Build encoding layers
        layers = []
        in_dim = state_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            
            if dropout > 0 and i < num_layers - 1:
                layers.append(nn.Dropout(dropout))
            
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
    

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """
        Encode particles to feature representations.
        
        Args:
            particles: Tensor of shape (batch, n_particles, state_dim)
            
        Returns:
            Encoded features of shape (batch, n_particles, hidden_dim)
        """
        batch_size, n_particles, state_dim = particles.shape
        
        # Flatten for processing
        flat_particles = particles.view(-1, state_dim)
        
        # Encode
        encoded = self.encoder(flat_particles)
        
        # Reshape back
        return encoded.view(batch_size, n_particles, self.hidden_dim)


class ParticleAttention(nn.Module):
    """
    Self-attention mechanism for particle interactions.
    
    This module enables particles to share information and
    capture multi-modal structure in the belief distribution.
    """
    
    def __init__(self,
                hidden_dim: int,
                num_heads: int = 4,
                dropout: float = 0.0):
        """
        Initialize particle attention.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(ParticleAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply self-attention to particle features.
        
        Args:
            x: Particle features of shape (batch, n_particles, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            Attended features of same shape
        """
        # Apply multi-head attention
        attended, attention_weights = self.attention(x, x, x, key_padding_mask=mask)
        
        # Residual connection
        x = x + attended
        
        # Output projection
        output = self.output_projection(x)
        
        return output


class BeliefAggregator(nn.Module):
    """
    Aggregates particle features into a belief representation.
    
    This module combines information from all particles to create
    a comprehensive belief representation for decision-making.
    """
    
    def __init__(self,
                hidden_dim: int,
                aggregation_method: str = 'attention_weighted',
                temperature: float = 1.0):
        """
        Initialize belief aggregator.
        
        Args:
            hidden_dim: Hidden dimension
            aggregation_method: Method for aggregation ('mean', 'attention_weighted', 'max')
            temperature: Temperature for attention weights
        """
        super(BeliefAggregator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.aggregation_method = aggregation_method
        self.temperature = temperature
        
        if aggregation_method == 'attention_weighted':
            # Learn aggregation weights
            self.weight_network = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
    

    def forward(self, particle_features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate particle features into belief representation.
        
        Args:
            particle_features: Features of shape (batch, n_particles, hidden_dim)
            
        Returns:
            Aggregated belief of shape (batch, hidden_dim)
        """
        if self.aggregation_method == 'mean':
            return torch.mean(particle_features, dim=1)
        
        elif self.aggregation_method == 'max':
            return torch.max(particle_features, dim=1)[0]
        
        elif self.aggregation_method == 'attention_weighted':
            # Compute attention weights
            weights = self.weight_network(particle_features)  # (batch, n_particles, 1)
            weights = weights.squeeze(-1) / self.temperature  # (batch, n_particles)
            weights = F.softmax(weights, dim=1)
            
            # Weighted aggregation
            weighted_features = particle_features * weights.unsqueeze(-1)
            return torch.sum(weighted_features, dim=1)
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")


class PolicyNetwork(nn.Module):
    """
    Main policy network for ESCORT that processes particle-based beliefs.
    
    This network implements the policy described in Section 3.5 of the paper,
    processing particles through encoding, attention, and aggregation stages
    to produce action distributions.
    """
    
    def __init__(self,
                state_dim: int,
                action_dim: int,
                hidden_dim: int = 128,
                num_encoder_layers: int = 2,
                num_heads: int = 4,
                discrete_actions: bool = True,
                aggregation_method: str = 'attention_weighted',
                dropout: float = 0.0,
                activation: str = 'relu'):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            num_encoder_layers: Number of particle encoder layers
            num_heads: Number of attention heads
            discrete_actions: Whether actions are discrete
            aggregation_method: Method for particle aggregation
            dropout: Dropout probability
            activation: Activation function ('relu', 'tanh', 'gelu')
        """
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.discrete_actions = discrete_actions
        
        # Particle encoder
        self.particle_encoder = ParticleEncoder(
            state_dim, 
            hidden_dim, 
            num_encoder_layers,
            dropout
        )
        
        # Particle attention
        self.particle_attention = ParticleAttention(
            hidden_dim,
            num_heads,
            dropout
        )
        
        # Belief aggregator
        self.belief_aggregator = BeliefAggregator(
            hidden_dim,
            aggregation_method
        )
        
        # Belief processor
        self.belief_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self._get_activation(activation)
        )
        
        # Output heads
        if discrete_actions:
            self.action_head = nn.Linear(hidden_dim, action_dim)
        else:
            # For continuous actions, output mean and log_std
            self.action_mean = nn.Linear(hidden_dim, action_dim)
            self.action_log_std = nn.Linear(hidden_dim, action_dim)
            
            # Initialize log_std to reasonable values
            nn.init.constant_(self.action_log_std.bias, -1.0)
    

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    

    def forward(self, 
                particles: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the policy network.
        
        Args:
            particles: Belief particles of shape (batch, n_particles, state_dim)
            mask: Optional particle mask of shape (batch, n_particles)
            
        Returns:
            For discrete actions: Action probabilities of shape (batch, action_dim)
            For continuous actions: Tuple of (mean, log_std) each of shape (batch, action_dim)
        """
        # Encode particles
        particle_features = self.particle_encoder(particles)
        
        # Apply self-attention
        attended_features = self.particle_attention(particle_features, mask)
        
        # Aggregate into belief representation
        belief_features = self.belief_aggregator(attended_features)
        
        # Process belief representation
        processed_belief = self.belief_processor(belief_features)
        
        # Generate action distribution
        if self.discrete_actions:
            action_logits = self.action_head(processed_belief)
            action_probs = F.softmax(action_logits, dim=-1)
            return action_probs
        else:
            action_mean = self.action_mean(processed_belief)
            action_log_std = self.action_log_std(processed_belief)
            
            # Clamp log_std for numerical stability
            action_log_std = torch.clamp(action_log_std, -20, 2)
            
            return action_mean, action_log_std
    

    def get_action(self,
                particles: torch.Tensor,
                deterministic: bool = False,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Get action from the policy.
        
        Args:
            particles: Belief particles
            deterministic: Whether to act deterministically
            mask: Optional particle mask
            
        Returns:
            Tuple of (action, info_dict)
        """
        with torch.no_grad():
            if self.discrete_actions:
                action_probs = self.forward(particles, mask)
                
                if deterministic:
                    action = torch.argmax(action_probs, dim=-1)
                    log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)))
                else:
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                
                info = {
                    'action_probs': action_probs,
                    'log_prob': log_prob,
                    'entropy': dist.entropy() if not deterministic else None
                }
            
            else:
                action_mean, action_log_std = self.forward(particles, mask)
                action_std = torch.exp(action_log_std)
                
                if deterministic:
                    action = action_mean
                    log_prob = None
                else:
                    dist = torch.distributions.Normal(action_mean, action_std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                
                info = {
                    'action_mean': action_mean,
                    'action_std': action_std,
                    'log_prob': log_prob,
                    'entropy': dist.entropy().sum(dim=-1) if not deterministic else None
                }
        
        return action, info
    

    def compute_log_prob(self,
                        particles: torch.Tensor,
                        actions: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute log probability of actions under the policy.
        
        Args:
            particles: Belief particles
            actions: Actions to evaluate
            mask: Optional particle mask
            
        Returns:
            Log probabilities
        """
        if self.discrete_actions:
            action_probs = self.forward(particles, mask)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
            return log_probs.squeeze(1)
        else:
            action_mean, action_log_std = self.forward(particles, mask)
            action_std = torch.exp(action_log_std)
            
            dist = torch.distributions.Normal(action_mean, action_std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            
            return log_probs
    

    def compute_entropy(self,
                    particles: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute policy entropy for regularization.
        
        Args:
            particles: Belief particles
            mask: Optional particle mask
            
        Returns:
            Entropy values
        """
        if self.discrete_actions:
            action_probs = self.forward(particles, mask)
            dist = torch.distributions.Categorical(action_probs)
            return dist.entropy()
        else:
            action_mean, action_log_std = self.forward(particles, mask)
            action_std = torch.exp(action_log_std)
            
            dist = torch.distributions.Normal(action_mean, action_std)
            return dist.entropy().sum(dim=-1)


class PolicyOptimizer:
    """
    Optimizer for the ESCORT policy network using policy gradient methods.
    """
    
    def __init__(self,
                policy_network: PolicyNetwork,
                learning_rate: float = 3e-4,
                entropy_coef: float = 0.01,
                value_coef: float = 0.5,
                max_grad_norm: float = 0.5,
                optimizer_type: str = 'adam'):
        """
        Initialize policy optimizer.
        
        Args:
            policy_network: The policy network to optimize
            learning_rate: Learning rate
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient (if using actor-critic)
            max_grad_norm: Maximum gradient norm for clipping
            optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
        """
        self.policy_network = policy_network
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Create optimizer
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                policy_network.parameters(),
                lr=learning_rate
            )
        elif optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                policy_network.parameters(),
                lr=learning_rate,
                weight_decay=1e-4
            )
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                policy_network.parameters(),
                lr=learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Loss tracking
        self.loss_history = []
    

    def compute_policy_loss(self,
                        particles: torch.Tensor,
                        actions: torch.Tensor,
                        advantages: torch.Tensor,
                        old_log_probs: Optional[torch.Tensor] = None,
                        clip_ratio: float = 0.2) -> Tuple[torch.Tensor, Dict]:
        """
        Compute policy gradient loss.
        
        Args:
            particles: Belief particles
            actions: Actions taken
            advantages: Advantage estimates
            old_log_probs: Old log probabilities (for PPO)
            clip_ratio: PPO clipping ratio
            
        Returns:
            Tuple of (loss, info_dict)
        """
        # Compute current log probabilities
        log_probs = self.policy_network.compute_log_prob(particles, actions)
        
        if old_log_probs is not None:
            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
            
            # Track clipping statistics
            clip_fraction = ((ratio - 1).abs() > clip_ratio).float().mean()
        else:
            # Simple policy gradient loss
            policy_loss = -(log_probs * advantages).mean()
            clip_fraction = torch.tensor(0.0)
        
        # Entropy regularization
        entropy = self.policy_network.compute_entropy(particles)
        entropy_loss = -self.entropy_coef * entropy.mean()
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        info = {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy': entropy.mean().item(),
            'clip_fraction': clip_fraction.item()
        }
        
        return total_loss, info
    
    
    def update(self,
            particles: torch.Tensor,
            actions: torch.Tensor,
            advantages: torch.Tensor,
            old_log_probs: Optional[torch.Tensor] = None) -> Dict:
        """
        Update the policy network.
        
        Args:
            particles: Belief particles
            actions: Actions taken
            advantages: Advantage estimates
            old_log_probs: Old log probabilities (for PPO)
            
        Returns:
            Dictionary of update statistics
        """
        # Compute loss
        loss, info = self.compute_policy_loss(
            particles, actions, advantages, old_log_probs
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            self.max_grad_norm
        )
        info['grad_norm'] = grad_norm.item()
        
        # Optimizer step
        self.optimizer.step()
        
        # Track loss
        self.loss_history.append(loss.item())
        info['total_loss'] = loss.item()
        
        return info
