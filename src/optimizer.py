"""DDPG optimization implementation for bioreactor process control.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

import os
import numpy as np
from tqdm import tqdm
from collections import deque
from typing import Optional, Tuple, Dict, List

from src.environment import BioreactorEnv
from src.agent import Agent



def optimizer(
    env: BioreactorEnv,
    agent: Agent,
    n_episodes: int = 1000,
    window: int = 100,
    max_t: int = 30,
    checkpoint_dir: Optional[str] = None,
    best_model_name: Optional[str] = None,
    final_model_name: Optional[str] = None,
    early_stop_patience: int = 200,
    min_improvement: float = 0.01
) -> Tuple[Dict, Dict]:
    """Deep Deterministic Policy Gradient optimization for bioreactor process control.

    Args:
        env (BioreactorEnv): Bioreactor environment object
        agent (Agent): DDPG agent object that implements the policy
        n_episodes (int): Maximum number of training episodes
        window (int): Window size for computing moving average score
        max_t (int): Maximum number of timesteps per episode
        checkpoint_dir (str, optional): Directory to save model checkpoints
        best_model_name (str, optional): Filename to save best performing model
        final_model_name (str, optional): Filename to save final model
        early_stop_patience (int): Number of episodes to wait for improvement before early stopping
        min_improvement (float): Minimum improvement threshold for early stopping

    Returns:
        tuple: Contains best_result (dict) and training_history (dict)
"""
    # create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize training history with deque for efficient memory usage
    training_history = {
        'scores': [],
        'actor_losses': [],
        'critic_losses': []
    }
    
    max_avg_score = -np.inf
    best_result = OptimizationResult()
    episodes_without_improvement = 0
    scores_window = deque(maxlen=window)
    
    # Main training loop with progress bar
    bar_format = (
        "🧬 Optimizing: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}]{postfix}"
    )
    with tqdm(range(1, n_episodes + 1), bar_format=bar_format, dynamic_ncols=True) as pbar:
        for i_episode in pbar:
            episode_data = run_episode(env, agent, max_t)
            
            # Update training history
            training_history['scores'].append(episode_data['total_reward'])
            training_history['critic_losses'].append(episode_data['critic_loss'])
            training_history['actor_losses'].append(episode_data['actor_loss'])
            scores_window.append(episode_data['total_reward'])

            # Update best results if we have improvement
            if best_result.update(i_episode, env, 
                                  episode_data['final_titer'], 
                                  episode_data['states'], 
                                  episode_data['actions']):
                agent.save(os.path.join(checkpoint_dir, best_model_name))

            # Check for new best average score
            if len(scores_window) == window:
                avg_score = np.mean(scores_window)
                if avg_score > max_avg_score + min_improvement:
                    max_avg_score = avg_score
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += 1

                # Update progress bar
                pbar.set_postfix({
                    'Avg Score': f'{avg_score:.2f}',
                    'Best Score': f'{max_avg_score:.2f}',
                    'Final Titer': f'{episode_data["final_titer"]:.2f}'
                })

                # Early stopping check
                if episodes_without_improvement >= early_stop_patience:
                    print(f"\nEarly stopping triggered after {i_episode} episodes")
                    break

    # Save final model
    agent.save(os.path.join(checkpoint_dir, final_model_name))
    
    best_result.format_print()
    return best_result.to_dict(), training_history


def run_episode(env: BioreactorEnv, agent: Agent, max_t: int) -> Dict:
    """Run a single episode of the optimization process."""
    state, _ = env.reset()
    agent.reset()
    
    states = []
    actions = []
    total_reward = 0
    total_critic_loss = 0
    total_actor_loss = 0
    
    for t in range(max_t):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        loss = agent.step(state, action, reward, next_state, done)
        
        states.append(state[:4])  # Only store VCD, Glucose, Lactate, Titer
        actions.append(action)
        
        total_reward += reward
        if done:
            break
            
        if loss:
            critic_loss, actor_loss = loss
            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            
        state = next_state
    
    return {
        'states': states,
        'actions': actions,
        'total_reward': total_reward,
        'critic_loss': total_critic_loss,
        'actor_loss': total_actor_loss,
        'final_titer': state[3]
    }

class OptimizationResult:
    """Class to store and manage optimization results."""
    def __init__(self):
        self.params = {}
        self.final_titer = -np.inf
        self.episode = 0
        self.predictions = None
        self.time_points = None
        
    def update(self, 
               episode: int, 
               env: BioreactorEnv, 
               final_titer: float,  
               states: List[np.ndarray], 
               actions: List[np.ndarray]) -> bool:
        """Update best results if current result is better."""
        if final_titer > self.final_titer:
            self.final_titer = final_titer
            self.episode = episode
            self.params = {
                'feed_start': env.current_params['feed_start'],
                'feed_end': env.current_params['feed_end'],
                'Glc_feed_rate': np.mean([a[0] for a in actions]),
                'Glc_0': env.current_params['Glc_0'],
                'VCD_0': env.current_params['VCD_0']
            }
            self.predictions = np.array(states)
            self.time_points = np.arange(env.config.TOTAL_DAYS)
            return True
        return False
    
    def to_dict(self) -> Dict:
        """Convert the optimization results to a dictionary."""
        return {
            'params': self.params,
            'final_titer': self.final_titer,
            'episode': self.episode,
            'predictions': self.predictions,
            'time_points': self.time_points
        }

    def format_print(self):
        """Print formatted optimization results."""
        print("Current optimal process conditions:")
        for param, value in self.params.items():
            print(f"{param}: {value:.4f}")
        print(f"Predicted final titer: {self.final_titer:.2f} mg/L")
        print(f"Found in episode: {self.episode}")