"""Hybrid model implementation combining mechanistic and ML approaches.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

import numpy as np
from tqdm import tqdm
from scipy.integrate import odeint


def ode_fcn(t, y, models, feeds=None, volumes=None, sign_mask=None):
    """
    --- Inputs ---
    t: Current timestep of the process
    y: Current states of VCD and Glucose
    feed: Feed rate for the experimentf
    g_mdl, k_mdl: Models for derivatives of growth and consumption rate
                  (sklearn expects 2D array, even when predicting on a single observation, use .reshape(1, -1))
    --- Outputs ---
    dVCD_dt, dGlc_df : Derivatives of VCD and Glucose for the next state from current one
    """
    # Current time index
    time_idx = int(t // 24)
    
    # Ensure time index does not exceed feed array bounds
    time_idx = min(time_idx, feeds.shape[0] - 1)

    # Get current feed rate
    current_feeds = feeds[time_idx]

    curr_volume = volumes[time_idx-1][0]
    after_feed_volume = volumes[time_idx][0]

    # mass balances
    dX_dt = np.zeros(len(models))
    for i, model in models.items():
        dX_dt[i] = (
            sign_mask[i] * model.predict(y.reshape(1, -1))[0] * curr_volume  
            + current_feeds[i] * after_feed_volume 
            - (y[i] * (after_feed_volume - curr_volume) / 24)
            ) / after_feed_volume

    return dX_dt


def simulate(
    models,
    sign_mask,
    init_conditions,
    time_points,
    F,
    V,
    Z,
):
    """Simulate system dynamics using a hybrid model.

    Args:
        model: Hybrid model
        init_conditions: Initial state values [batch_size, num_states]
        time_points: Array of time points [num_time_points]
        F: Feeding rate data [batch_size, time_steps, num_feeds]
        V: Volume data [batch_size, time_steps, 1]
        Z: DOE data [batch_size, time_steps, num_doe_params]

    Returns:
        torch.Tensor: Simulated state trajectories
    """
    batch_size = init_conditions.shape[0]
    num_time_points = len(time_points)
    num_states = init_conditions.shape[1]

    X_pred = np.zeros(
        (batch_size, num_time_points, num_states)
    )

    for i in tqdm(range(batch_size), desc="Simulating: ", total=batch_size, leave=False, ncols=80):
        # Initial Values
        init_values = init_conditions[i]

        # Get all the time-steps on which we want to predict
        t_eval = time_points

        # Get feed rate data
        feeds = F[i, :, :]

        volumes = V[i, :, :]

        X_pred[i, :, :] = odeint(
            func=ode_fcn,
            y0=init_values,
            t=t_eval,
            args=(models, feeds, volumes, sign_mask),
            tfirst=True,
        )  # T, C

    return X_pred


def save_hybrid_model(models, sign_mask, save_path='hybrid_model.pkl'):
    """
    Save hybrid model and its components.
    
    Args:
        models: Dictionary of trained GP models
        sign_mask: Array of signs for each state variable
        save_path: Path to save the model file
    """
    import pickle
    model_data = {
        'models': models,
        'sign_mask': sign_mask
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {save_path}")


def load_hybrid_model(load_path='hybrid_model.pkl'):
    """
    Load hybrid model and its components.
    
    Args:
        load_path: Path to the saved model file
    
    Returns:
        models: Dictionary of trained GP models
        sign_mask: Array of signs for each state variable
    """
    import pickle
    
    with open(load_path, 'rb') as f:
        model_data = pickle.load(f)
    
    models = model_data['models']
    sign_mask = model_data['sign_mask']
    
    print(f"Model loaded from {load_path}")
    return models, sign_mask