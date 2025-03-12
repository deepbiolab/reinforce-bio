from typing import Dict, Any

def get_default_config() -> Dict[str, Any]:
    """Get default configuration for the bioreactor system.
    
    Returns:
        Dict[str, Any]: Default configuration dictionary containing all parameters
    """
    return {
        "results_dir": "results",  # Directory for saving results
        "random_seed": 42,  # Random seed for reproducibility
        
        # Hybrid model configuration
        "hybrid_model_params": {
            "mode": "train",  # Operation mode: 'train', 'test', or 'predict'
            "checkpoint_dir": "checkpoints",  # Directory for saving model checkpoints
            "visualization": {
                "select_runs": [20, 21, 22, 23, 24]  # Default runs to visualize
            },
            "model_path": "checkpoints/hybrid_model.pkl",
        },
        
        # Dataset configuration parameters
        "dataset_params": {
            "owu_file": "owu",  # Observation Wise Unit data file
            "doe_file": "owu_doe",  # Design of Experiments file
            "train_path": "dataset/interpolation/train",  # Training data path
            "test_path": "dataset/interpolation/test",    # Test data path
            "predict_path": "dataset/interpolation/predict",  # Prediction data path
            "Z_columns": ["feed_start", "feed_end", "Glc_feed_rate", "Glc_0", "VCD_0"], # Design parameters
            "F_columns": ["Glc"],  # Feeding variables (Glucose)
            "X_columns": ["VCD", "Glc", "Lac", "Titer"], # States
            "t_steps": 15,  # Number of time steps
            "time_step": 24,  # Hours per time step
            "init_volume": 1000,  # Initial volume in mL
            "val_split": 0.2,  # Validation set ratio
        },
        
        # Environment configuration parameters
        "env_params": {
            # Process constraints configuration
            "constraints": {
                "MIN_GLUCOSE": 5.0,
                "MAX_GLUCOSE": 80.0,
                "LOW_GLUCOSE_PENALTY": -10.0,
                "HIGH_GLUCOSE_PENALTY": -5.0,
            },
            # Process configuration
            "process": {
                "INIT_VOLUME_ML": 1000.0,
                "TIME_STEP_HOURS": 24,
                "TOTAL_DAYS": 15,
                "MAX_VCD": 100.0,
                "MAX_GLUCOSE": 100.0,
                "MAX_LACTATE": 100.0,
                "MAX_TITER": 1000.0,
            },
            # Parameter ranges
            "param_ranges": {
                "feed_start": (1.0, 5.0),
                "feed_end": (8.0, 12.0),
                "Glc_feed_rate": (5.0, 20.0),
                "VCD_0": (0.5, 1.1),
                "Glc_0": (30.0, 75.0),
            },
        },
        
        # Agent configuration parameters
        "agent_params": {
            "test_mode": False,
            "hidden_size": (400, 300),
            "buffer_size": int(1e6),
            "batch_size": 64,
            "gamma": 0.99,
            "tau": 1e-3,
            "lr_actor": 1e-4,
            "lr_critic": 1e-3,
            "seed": 42,
            "device": "mps"
        },
        
        # DDPG training parameters
        "ddpg_params": {
            "n_episodes": 10, # 300
            "window": 100,
            "max_t": 300,
            "checkpoint_dir": "checkpoints",  # Directory for saving agent weights
            "best_model_name": "best_agent.pth",  # Filename for best model
            "final_model_name": "final_agent.pth",  # Filename for final model
        },
        
        # Plotting parameters
        "plot_params": {
            "rolling_window": 100,
        }
    }