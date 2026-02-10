from collections import deque
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np


class FlickeringAtariWrapper(gym.Wrapper):
    """
        Implements flickering mechanism for Atari games.

        This wrapper modifies standard Atari environments to create partial observability by randomly replacing frames with blank frames.

        Args:
            env: Base gymnasium Atari environment (should be NoFrameskip version)
            flicker_prob: Probability of replacing frame with blank (default: 0.5)
            action_repeat_prob: Probability of repeating action twice (default: 0.25)
            frameskip: Number of frames to skip (default: 4)
    """
    def __init__(
        self, 
        env: gym.Env,
        flicker_prob: float = 0.5,
        action_repeat_prob: float = 0.25,
        frameskip: int = 4
    ):
        """
            Initialize flickering wrapper with frame corruption parameters.
        """
        assert 0 <= flicker_prob <= 1, "flicker_prob must be between 0 and 1"
        assert 0 <= action_repeat_prob <= 1, "action_repeat_prob must be between 0 and 1"
        assert frameskip > 0, "frameskip must be positive"

        super().__init__(env)

        self.flicker_prob = flicker_prob
        self.action_repeat_prob = action_repeat_prob
        self.frameskip = frameskip

        # & Create blank frame (all zeros) matching observation space
        self.blank_frame = np.zeros(
            self.observation_space.shape, 
            dtype=self.observation_space.dtype
        )

        # & Frame buffer for max pooling (standard Atari preprocessing)
        self.frame_buffer = deque(maxlen=2)

        # & Statistics tracking
        self._episode_stats = {
            'frames_flickered': 0,
            'actions_repeated': 0,
            'total_frames': 0,
            'total_actions': 0
        }

    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
            Reset environment and apply flickering to initial observation.

            Returns:
                observation: Initial observation (possibly flickered)
                info: Dictionary with reset info and flickering status
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # & Clear frame buffer
        self.frame_buffer.clear()

        # & Reset episode statistics
        self._episode_stats = {
            'frames_flickered': 0,
            'actions_repeated': 0,
            'total_frames': 0,
            'total_actions': 0
        }

        # & Apply flickering to initial observation
        if self.np_random.random() < self.flicker_prob:
            obs = self.blank_frame.copy()
            info['flickered'] = True
            self._episode_stats['frames_flickered'] += 1
        else:
            info['flickered'] = False
            
        self._episode_stats['total_frames'] += 1
        
        # & Add flickering rate to info
        if self._episode_stats['total_frames'] > 0:
            info['flickering_rate'] = (
                self._episode_stats['frames_flickered'] / 
                self._episode_stats['total_frames']
            )
        
        info['episode_stats'] = self._episode_stats.copy()
        
        return obs, info
    

    def step(
        self, 
        action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
            Execute action with flickering mechanism.

            1. 25% chance to repeat the action (execute twice)
            2. Apply frameskip of 4
            3. Max pool over last 2 frames
            4. 50% chance to replace final observation with blank frame
        """
        self._episode_stats['total_actions'] += 1

        # & Determine if we should repeat this action
        repeat_action = self.np_random.random() < self.action_repeat_prob

        if repeat_action:
            # & Execute action twice with flickering mechanism
            self._episode_stats['actions_repeated'] += 1
            
            # & First execution
            obs1, reward1, terminated1, truncated1, info1 = self._execute_action_with_frameskip(action)
            
            if terminated1 or truncated1:
                # & Episode ended on first execution
                info1['action_repeated'] = True
                info1['episode_stats'] = self._episode_stats.copy()
                return obs1, reward1, terminated1, truncated1, info1
            
            # & Second execution
            obs2, reward2, terminated2, truncated2, info2 = self._execute_action_with_frameskip(action)
            
            # & Combine results
            total_reward = reward1 + reward2
            info2['action_repeated'] = True
            info2['episode_stats'] = self._episode_stats.copy()
            
            return obs2, total_reward, terminated2, truncated2, info2
        else:
            # & Normal single execution
            obs, reward, terminated, truncated, info = self._execute_action_with_frameskip(action)
            info['action_repeated'] = False
            info['episode_stats'] = self._episode_stats.copy()
            
            return obs, reward, terminated, truncated, info
        

    def _execute_action_with_frameskip(
        self, 
        action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
            Execute action for 'frameskip' frames and apply flickering.
            
            This follows standard Atari preprocessing:
            1. Repeat action for 'frameskip' frames
            2. Max pool over last 2 frames
            3. Apply flickering to final observation
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        
        # & Execute action for 'frameskip' frames
        for frame_idx in range(self.frameskip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            # & Store last 2 frames for max pooling
            # & This is standard Atari preprocessing to handle sprite flickering
            if frame_idx >= self.frameskip - 2:
                self.frame_buffer.append(obs)
            
            # & Stop if episode ended
            if terminated or truncated:
                break
        
        # & Apply max pooling over last 2 frames if we have them
        # & This helps with the natural flickering in Atari games
        if len(self.frame_buffer) == 2:
            obs = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        
        # & Apply flickering: 50% chance to replace with blank frame
        self._episode_stats['total_frames'] += 1
        
        if self.np_random.random() < self.flicker_prob:
            obs = self.blank_frame.copy()
            info['flickered'] = True
            self._episode_stats['frames_flickered'] += 1
        else:
            info['flickered'] = False
        
        # & Add flickering rate to info
        if self._episode_stats['total_frames'] > 0:
            info['flickering_rate'] = (
                self._episode_stats['frames_flickered'] / 
                self._episode_stats['total_frames']
            )
        
        return obs, total_reward, terminated, truncated, info
    

    def close(self):
        """
            Close the environment.
        """
        return self.env.close()
    

    def __str__(self):
        """
            String representation of the wrapper.
        """
        return (
            f"<FlickeringAtariWrapper("
            f"env={self.env.spec.id if self.env.spec else 'Unknown'}, "
            f"flicker_prob={self.flicker_prob}, "
            f"action_repeat_prob={self.action_repeat_prob}, "
            f"frameskip={self.frameskip})>"
        )

