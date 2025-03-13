"""Bioreactor environment for DDPG optimization.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.integrate import odeint
from typing import Dict, Tuple, Any, List
from dataclasses import dataclass
from enum import IntEnum


@dataclass
class ProcessConstraints:
    """Process constraints configuration."""

    MIN_GLUCOSE: float = 5.0
    MAX_GLUCOSE: float = 80.0

    # Penalties for constraint violations
    LOW_GLUCOSE_PENALTY: float = -10.0
    HIGH_GLUCOSE_PENALTY: float = -5.0


@dataclass
class ProcessConfig:
    """Process configuration parameters."""

    INIT_VOLUME_ML: float = 1000.0
    TIME_STEP_HOURS: int = 24
    TOTAL_DAYS: int = 15

    # State variable ranges (with units)
    MAX_VCD: float = 50.0      # 10^6 cells/mL
    MAX_GLUCOSE: float = 80.0  # g/L
    MAX_LACTATE: float = 5.4   # g/L
    MAX_TITER: float = 5000.0  # mg/L


class StateIndex(IntEnum):
    """Indices for state variables."""

    VCD = 0
    GLUCOSE = 1
    LACTATE = 2
    TITER = 3


class ParamConfig:
    """Parameter ranges and default values."""

    RANGES = {
        "feed_start": (1.0, 5.0),
        "feed_end": (8.0, 12.0),
        "Glc_feed_rate": (5.0, 20.0),
        "VCD_0": (0.5, 1.1),
        "Glc_0": (30.0, 75.0),
    }

    @classmethod
    def get_default_ranges(cls) -> Dict[str, Tuple[float, float]]:
        return cls.RANGES.copy()


class BioreactorEnv(gym.Env):
    """Custom Environment that follows gym interface for bioreactor optimization."""

    def __init__(
        self,
        models: Dict,
        sign_mask: np.ndarray,
        constraints: Dict = None,
        process: Dict = None,
        param_ranges: Dict = None,
    ):
        """Initialize BioreactorEnv.

        Args:
            models: Dictionary of trained models for each state variable
            sign_mask: Array of signs for each state variable
            param_ranges: Dictionary of parameter ranges
            process_config: Process configuration parameters
            process_constraints: Process constraints configuration
        """
        super().__init__()

        # Create environment configurations from config
        self.config = ProcessConfig(**process)
        self.constraints = ProcessConstraints(**constraints)
        self.param_ranges = param_ranges or ParamConfig.get_default_ranges()

        self.models = models
        self.sign_mask = sign_mask

        # Initialize spaces
        self._setup_spaces()

        # Initialize state arrays
        self._initialize_state_arrays()

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        # Sample new initial parameters
        self.current_params = self._sample_initial_params()

        # Initialize state
        self.current_state = np.zeros(self.state_dim)
        self.current_state[StateIndex.VCD] = self.current_params["VCD_0"]
        self.current_state[StateIndex.GLUCOSE] = self.current_params["Glc_0"]

        # Reset time step and volume
        self.current_step = 0
        self.V[0, 0, 0] = self.config.INIT_VOLUME_ML / 1000  # Convert to L

        # Create observation
        obs = self._get_observation()

        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        # Update feed rate and volume
        self._update_process_conditions(action)

        # Simulate one step
        next_state = self._simulate_step()

        # Calculate reward
        reward = self._calculate_reward(next_state)

        # Update state and time
        self.current_state = next_state
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= self.config.TOTAL_DAYS

        # Create observation
        obs = self._get_observation()

        return obs, reward, done, False, {}
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Action space: Glucose feed rate for each day
        self.action_space = spaces.Box(
            low=np.float32(self.param_ranges["Glc_feed_rate"][0]),
            high=np.float32(self.param_ranges["Glc_feed_rate"][1]),
            shape=(1,),
            dtype=np.float32,
        )

        # Observation space: [VCD, Glucose, Lactate, Titer]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array(
                [
                    self.config.MAX_VCD,
                    self.config.MAX_GLUCOSE,
                    self.config.MAX_LACTATE,
                    self.config.MAX_TITER,
                ],
                dtype=np.float32
            ),
            dtype=np.float32,
        )
    
    def _get_observation(self) -> np.ndarray:
        """Create observation vector from current state."""
        return self.current_state
    
    def _initialize_state_arrays(self):
        """Initialize arrays for storing process data."""
        self.state_dim = len(StateIndex)

        self.X = np.zeros((1, self.config.TOTAL_DAYS, self.state_dim))  # States
        self.F = np.zeros((1, self.config.TOTAL_DAYS, self.state_dim))  # Feeding rates
        self.Z = np.zeros((1, self.config.TOTAL_DAYS, len(self.param_ranges)))  # Parameters
        self.V = np.zeros((1, self.config.TOTAL_DAYS, 1))  # Volume

    def _sample_initial_params(self) -> Dict[str, float]:
        """Sample initial parameters from their ranges."""
        return {
            param: np.random.uniform(low, high)
            for param, (low, high) in self.param_ranges.items()
            if param != "Glc_feed_rate"  # Exclude feed rate as it's the action
        }

    def _update_process_conditions(self, action: np.ndarray):
        """Update process conditions based on action."""
        # Set feed rate: from [-1, 1](see agent.py, select_action method) into real range
        feed_range = self.param_ranges["Glc_feed_rate"]
        actual_feed = (action[0] + 1) * (feed_range[1] - feed_range[0]) / 2 + feed_range[0]
        
        hourly_rate = actual_feed / self.config.TIME_STEP_HOURS
        self.F[0, self.current_step, StateIndex.GLUCOSE] = hourly_rate

        # Update volume
        if self.current_step > 0:
            prev_volume = self.V[0, self.current_step - 1, 0]
            feed_volume = (
                np.sum(self.F[0, self.current_step]) * self.config.TIME_STEP_HOURS
            )
            self.V[0, self.current_step, 0] = prev_volume + feed_volume / 1000

    def _simulate_step(self) -> np.ndarray:
        """Simulate one time step using the hybrid model."""
        t_eval = np.array([0, self.config.TIME_STEP_HOURS])

        next_state = odeint(
            func=self._ode_fcn,
            y0=self.current_state,
            t=t_eval,
            args=(
                self.models,
                self.F[0, self.current_step],
                self.V[0, self.current_step],
                self.sign_mask,
            ),
            tfirst=True,
        )

        return next_state[-1]

    def _ode_fcn(self, t, y, models, feeds, volume, sign_mask):
        """ODE function for simulation."""
        dX_dt = np.zeros(len(models))

        for i, model in models.items():
            dX_dt[i] = (
                sign_mask[i] * model.predict(y.reshape(1, -1))[0] * volume[0]
                + feeds[i] * volume[0]
            ) / volume[0]

        return dX_dt

    def _calculate_reward(self, next_state: np.ndarray) -> float:
        """Calculate reward based on titer increase and constraints."""
        # Calculate titer increase
        titer_increase = (
            next_state[StateIndex.TITER] - self.current_state[StateIndex.TITER]
        )

        # Apply glucose constraints
        glucose_level = next_state[StateIndex.GLUCOSE]
        glucose_penalty = 0.0

        if glucose_level < self.constraints.MIN_GLUCOSE:
            glucose_penalty = self.constraints.LOW_GLUCOSE_PENALTY
        elif glucose_level > self.constraints.MAX_GLUCOSE:
            glucose_penalty = self.constraints.HIGH_GLUCOSE_PENALTY

        return titer_increase + glucose_penalty
