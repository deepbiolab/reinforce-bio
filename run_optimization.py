"""
Main script for running bioreactor optimization using hybrid model and DDPG.

This script loads the configuration, dataset and models, then runs the optimization
process and generates visualization plots for the results.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

from src.config import get_default_config
from src.dataset import BioreactorDataset
from src.hybrid_model import load_hybrid_model
from src.environment import BioreactorEnv
from src.agent import Agent
from src.optimizer import optimizer
from src.plots import plot_optimal_conditions, plot_scores


def main():
    # Get configuration
    config = get_default_config()

    bioreactor_dataset = BioreactorDataset(mode='train', **config['dataset_params'])

    models, sign_mask = load_hybrid_model(config['hybrid_model_params']['model_path'])

    # Create environment with all configurations
    env = BioreactorEnv(models=models, sign_mask=sign_mask, **config['env_params'])

    # Create agent with config parameters
    agent = Agent(env, **config['agent_params'])

    # Run DDPG optimization with config parameters
    optimal_result, training_history = optimizer(env, agent, **config['ddpg_params'])

    # Plot training scores with config parameters
    rolling_mean = plot_scores(
        scores=training_history["scores"],
        actor_losses=training_history["actor_losses"],
        critic_losses=training_history["critic_losses"],
        rolling_window=config['plot_params']['rolling_window']
    )

    # Plot the predicted profiles for the optimal conditions
    plot_optimal_conditions(optimal_result, bioreactor_dataset.X_columns)

if __name__ == "__main__":
    main()