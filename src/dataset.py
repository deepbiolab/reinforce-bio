"""Bioreactor dataset implementation.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

import os
from typing import Tuple, List
import numpy as np
import pandas as pd


def process_owu_data(
    owu_raw: pd.DataFrame, t_steps: int, X_columns: List[str], F_columns: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert OWU DataFrame to 3D numpy arrays.

    Args:
        owu_raw: Raw OWU DataFrame
        t_steps: Number of time steps
        X_columns: List of state variable column names
        F_columns: List of feeding variable column names

    Returns:
        Tuple containing:
        - State variables array [batch_size, time_steps, num_vars]
        - Feeding rates array [batch_size, time_steps, num_feeds]
    """
    owu = owu_raw.copy()
    owu = owu.sort_index(level=["run", "time"])

    B = owu.index.get_level_values("run").nunique()
    T = t_steps
    C_X = len(X_columns)
    C_F = len(F_columns)

    X = np.zeros((B, T, C_X))
    F = np.zeros((B, T, C_F))

    for i, (run, group) in enumerate(owu.groupby(level="run")):
        X_group = group[X_columns].copy()
        F_group = group[F_columns].copy()

        if len(group) != T:
            raise ValueError(f"Run {run} does not have {T} time steps.")

        X[i, :, :] = X_group.values
        F[i, :, :] = F_group.values

    return X, F


def process_doe_data(doe_raw: pd.DataFrame, Z_columns: List[str]) -> np.ndarray:
    """Convert DOE DataFrame to 3D numpy array.

    Args:
        doe_raw: Raw DOE DataFrame
        Z_columns: List of DOE parameter column names

    Returns:
        np.ndarray: 3D array of experimental parameters [batch_size, 1, num_params]
    """
    doe = doe_raw.copy()
    doe = doe.sort_index()

    B = doe.shape[0]
    C_Z = len(Z_columns)
    T = 1

    Z = np.zeros((B, T, C_Z))
    Z[:, 0, :] = doe.values

    return Z


def flatten_dataset(X, F, Y):
    B, T, C = X.shape
    X_flat = X.reshape(B * T, C)
    Y_flat = Y.reshape(B * T, C)
    F_flat = F.reshape(B * T, C)

    return X_flat, F_flat, Y_flat


def create_empty_owu(
    file: str,
    doe_data: pd.DataFrame,
    t_steps: int,
    F_columns: List[str],
    X_columns: List[str],
    root_path: str,
) -> pd.DataFrame:
    """Create an OWU data framework pre-filled with feeding information.

    Args:
        file: OWU file name
        doe_data: DOE DataFrame containing feed parameters
        t_steps: Number of time steps
        F_columns: List of feeding variable column names
        X_columns: List of state variable column names
        root_path: Root path for data files

    Returns:
        pd.DataFrame: Pre-filled OWU data framework
    """
    header_df = pd.read_csv(f"{root_path}/{file}.csv", nrows=0)

    index = pd.MultiIndex.from_product(
        [list(range(doe_data.shape[0])), list(range(t_steps))],
        names=["run", "time"],
    )

    empty_df = pd.DataFrame(0.0, index=index, columns=header_df.columns)

    for run in range(doe_data.shape[0]):
        feed_start = int(doe_data.iloc[run]["feed_start"])
        feed_end = int(doe_data.iloc[run]["feed_end"])
        feed_rate = doe_data.iloc[run]["Glc_feed_rate"]

        for time in range(feed_start, min(feed_end + 1, t_steps)):
            for col in F_columns:
                empty_df.loc[(run, time), col] = feed_rate

    empty_df = empty_df[X_columns + F_columns]
    return empty_df


class BioreactorDataset:
    """Dataset class for bioreactor data handling and processing."""

    def __init__(
        self,
        owu_file: str,
        doe_file: str,
        train_path: str = "dataset/interpolation/train",
        test_path: str = "dataset/interpolation/test",
        predict_path: str = "dataset/interpolation/predict",
        t_steps: int = 15,
        time_step: int = 24,
        init_volume: float = 1000,
        Z_columns: List[str] = [],
        X_columns: List[str] = [],
        F_columns: List[str] = [],
        mode: str = "train",
        val_split: float = 0.2,
        random_seed: int = 42,
    ) -> None:
        """Initialize the dataset.

        Args:
            owu_file: Name of the OWU data file
            doe_file: Name of the DOE data file
            train_path: Root path for training dataset
            test_path: Root path for test dataset
            predict_path: Root path for prediction dataset
            t_steps: Number of time steps
            time_step: Duration of each time step in hours
            init_volume: Initial volume in mL
            Z_columns: List of DOE parameter columns
            X_columns: List of state variable columns
            F_columns: List of feeding rate columns
            mode: Dataset mode ('train', 'val', 'test', or 'predict')
            val_split: Validation set ratio (0-1)
            random_seed: Random seed for reproducibility
        """
        self.t_steps = t_steps
        self.time_step = time_step
        self.init_volume = init_volume
        self.mode = mode
        self.train_path = train_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.val_split = val_split
        self.random_seed = random_seed

        self.time_mask = np.ones(t_steps)
        self.sign_mask = np.array([-1 if col in F_columns else 1 for col in X_columns])
        self.feed_mask = np.where(self.sign_mask < 0, 1, 0)

        self.Z_columns = Z_columns
        self.X_columns = [f"X:{col}" for col in X_columns]
        self.F_columns = [f"F:{col}" for col in F_columns]

        if mode == "train" or mode == "val":
            self.root_path = train_path
        elif mode == "test":
            self.root_path = test_path
        elif mode == "predict":
            self.root_path = predict_path
        else:
            raise ValueError("Invalid mode. Choose from 'train', 'test', 'predict'.")

        doe_data = self._read_doe(doe_file)

        if mode == "predict":
            owu_data = create_empty_owu(
                owu_file,
                doe_data,
                t_steps,
                self.F_columns,
                self.X_columns,
                self.root_path,
            )
            self.result_df = owu_data.reset_index().copy()
        else:
            owu_data = self._read_owu(owu_file)

        self._process_data(owu_data, doe_data)

        # if mode in ["train", "val"]:
        #     self._split_data()

        self._init_conditions = None

    @classmethod
    def train_val_split(
        cls, **kwargs
    ) -> Tuple["BioreactorDataset", "BioreactorDataset"]:
        """Create training and validation dataset instances."""
        train_dataset = cls(mode="train", **kwargs)
        val_dataset = cls(mode="val", **kwargs)
        return train_dataset, val_dataset

    def _split_data(self) -> None:
        """Split the data into training and validation sets."""
        np.random.seed(self.random_seed)
        total_size = len(self.X)
        val_size = int(self.val_split * total_size)
        indices = list(range(total_size))
        np.random.shuffle(indices)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        self.indices = train_indices if self.mode == "train" else val_indices
        self.X = self.X[self.indices]
        self.F = self.F[self.indices]
        self.Y = self.Y[self.indices]
        self.V = self.V[self.indices]
        self.Z = self.Z[self.indices]

    def _read_owu(self, file: str) -> pd.DataFrame:
        """Read and process the OWU data file."""
        data = pd.read_csv(f"{self.root_path}/{file}.csv")
        owu_df = data.copy()
        num_runs = len(pd.read_csv(f"{self.root_path}/{file}_doe.csv"))

        if "run" not in owu_df.columns:
            owu_df.index = pd.MultiIndex.from_product(
                [list(range(num_runs)), list(range(self.t_steps))],
                names=["run", "time"],
            )
        else:
            owu_df.set_index(["run", "time"], inplace=True)
        owu_df = owu_df[self.X_columns + self.F_columns]
        return owu_df

    def _read_doe(self, file: str) -> pd.DataFrame:
        """Read the Design of Experiments data file."""
        data = pd.read_csv(
            f"{self.root_path}/{file}.csv",
            usecols=self.Z_columns,
        )
        return data.copy()

    def _process_data(self, owu_data: pd.DataFrame, doe_data: pd.DataFrame) -> None:
        """Process raw data into tensor format."""
        self.X, self.F = process_owu_data(
            owu_data, self.t_steps, self.X_columns, self.F_columns
        )
        self.Z = process_doe_data(doe_data, self.Z_columns)

        self.V = (
            self.init_volume + (self.F.sum(axis=-1, keepdims=True)).cumsum(axis=1)
        ) / 1000

        self.F = (
            self.feed_mask[None, None, :] * self.F
        ) / self.time_step  # [1, 1, C] x [B, T, 1] -> [B, T, C]
        self.Z = (
            self.time_mask[None, :, None] * self.Z
        )  # [1, T, 1] x [B, 1, C] -> [B, T, C]

        if self.mode != "predict":
            self.Y = self._central_differences(self.X, self.F, self.V)
        else:
            self.Y = np.zeros_like(self.X)

    def _central_differences(
        self, X: np.ndarray, F: np.ndarray, V: np.ndarray
    ) -> np.ndarray:
        """Calculate central differences for derivatives computation."""
        Y = np.zeros_like(X)

        Y[:, 0, :] = (
            self.sign_mask[None, :]
            * (X[:, 1, :] * V[:, 1, :] - X[:, 0, :] * V[:, 0, :])
        ) / (self.time_step * V[:, 0, :]) + F[:, 0, :]

        Y[:, 1:-1, :] = (
            self.sign_mask[None, None, :]
            * ((X[:, 2:, :] * V[:, 2:, :] - X[:, :-2, :] * V[:, :-2, :]) / 2)
        ) / (self.time_step * V[:, 1:-1, :]) + F[:, 1:-1, :]

        Y[:, -1, :] = (
            self.sign_mask[None, :]
            * (X[:, -1, :] * V[:, -1, :] - X[:, -2, :] * V[:, -2, :])
        ) / (self.time_step * V[:, -1, :]) + F[:, -2, :]

        return Y

    @property
    def features_dim(self) -> int:
        """Total dimension of feature space."""
        return self.X.shape[-1] + self.F.shape[-1] + self.Z.shape[-1]

    @property
    def states_dim(self) -> int:
        """Dimension of state space."""
        return self.X.shape[-1]

    @property
    def init_conditions(self) -> np.array:
        """Get initial conditions tensor."""
        if self._init_conditions is None:
            self._init_conditions = np.zeros((len(self), self.states_dim))

            for i, col in enumerate(self.Z_columns):
                if col.endswith("_0") and f"X:{col[:-2]}" in self.X_columns:
                    state_idx = self.X_columns.index(f"X:{col[:-2]}")
                    self._init_conditions[:, state_idx] = self.Z[:, 0, i]

        return self._init_conditions

    def get_simulation_data(self) -> dict:
        """Get all data needed for simulation."""
        return {
            "init_conditions": self.init_conditions,
            "F": self.F,
            "V": self.V,
            "Z": self.Z,
            "time_points": np.arange(0, self.t_steps * self.time_step, self.time_step),
        }

    def save_predictions(self, X_pred: np.array, save_dir: str = "results") -> None:
        """Save prediction results."""
        os.makedirs(save_dir, exist_ok=True)
        self.result_df[self.X_columns] = X_pred.reshape(-1, X_pred.shape[-1])
        file_path = os.path.join(save_dir, "owu_pred.csv")
        self.result_df.to_csv(file_path, index=False)
        print(f"Predictions saved to {file_path}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
        """Get a single sample from the dataset."""
        if self.mode == "predict":
            return (self.F[idx], self.V[idx], self.Z[idx])
        else:
            return (self.X[idx], self.F[idx], self.Y[idx], self.V[idx], self.Z[idx])


