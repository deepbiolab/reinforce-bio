import os
import argparse
import numpy as np
from tqdm import tqdm

from src.config import get_default_config
from src.dataset import BioreactorDataset, flatten_dataset
from src.gp_model import fit_gp_model
from src.hybrid_model import simulate, load_hybrid_model, save_hybrid_model
from src.metrics import r2, relative_rmse
from src.plots import plot_predicted_profile


def load_datasets(config, mode="train"):
    """Load and prepare datasets based on mode.
    
    Args:
        config: Configuration dictionary
        mode: Operation mode ('train', 'test', or 'predict')
    
    Returns:
        tuple: Dataset information depending on mode
    """
    if mode == "train":
        # Load both training and test datasets for training mode
        train_dataset = BioreactorDataset(mode="train", **config["dataset_params"])
        test_dataset = BioreactorDataset(mode="test", **config["dataset_params"])
        
        X_columns = train_dataset.X_columns
        sign_mask = train_dataset.sign_mask
        
        # Extract train data
        X_train = train_dataset.X
        Z_train = train_dataset.Z
        F_train = train_dataset.F
        Y_train = train_dataset.Y
        
        # Flatten training datasets
        X_flat, F_flat, Y_flat = flatten_dataset(X_train, F_train, Y_train)
        
        # Extract test data
        X_test = test_dataset.X
        
        return {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "X_columns": X_columns,
            "sign_mask": sign_mask,
            "X_train": X_train,
            "X_test": X_test,
            "X_flat": X_flat,
            "Y_flat": Y_flat
        }
        
    elif mode == "test":
        # Load only test dataset for test mode
        test_dataset = BioreactorDataset(mode="test", **config["dataset_params"])
        
        return {
            "test_dataset": test_dataset,
            "X_columns": test_dataset.X_columns,
            "sign_mask": test_dataset.sign_mask,
            "X_test": test_dataset.X
        }
        
    else:  # predict mode
        # Load only prediction dataset
        predict_dataset = BioreactorDataset(mode="predict", **config["dataset_params"])
        
        return {
            "predict_dataset": predict_dataset,
            "X_columns": predict_dataset.X_columns,
            "sign_mask": predict_dataset.sign_mask
        }



def train_models(X_flat, Y_flat, X_columns):
    """Train GP models for each variable"""
    models = {}
    for i, var_name in tqdm(
        enumerate(X_columns),
        desc="Training: ",
        total=len(X_columns),
        leave=True,
        ncols=80,
    ):
        model = fit_gp_model(X_flat, Y_flat[:, i])
        models[i] = model
    return models


def save_and_load_models(models, sign_mask, config):
    """Save and reload the trained models"""
    checkpoint_dir = config["hybrid_model_params"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "hybrid_model.pkl")
    save_hybrid_model(models, sign_mask, save_path=save_path)
    return load_hybrid_model(load_path=save_path)


def load_saved_models(config):
    """Load previously saved models"""
    checkpoint_dir = config["hybrid_model_params"]["checkpoint_dir"]
    load_path = os.path.join(checkpoint_dir, "hybrid_model.pkl")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No saved model found at {load_path}")
    return load_hybrid_model(load_path=load_path)


def evaluate_predictions(X_true, X_pred, X_columns, dataset_name=""):
    """Evaluate model predictions and print metrics"""
    print(f"\n{dataset_name} Set")
    for i, var in enumerate(X_columns):
        rmse = relative_rmse(X_true[:, :, i], X_pred[:, :, i])
        score = r2(X_true[:, :, i], X_pred[:, :, i])
        print(f"{var} {dataset_name} RMSE: {rmse}, R2: {score}")


def train_pipeline(config):
    """Execute training pipeline"""
    # Load and prepare datasets
    data = load_datasets(config, mode="train")
    
    # Train models
    models = train_models(data["X_flat"], data["Y_flat"], data["X_columns"])
    
    # Save and reload models
    models, sign_mask = save_and_load_models(models, data["sign_mask"], config)
    
    # Generate predictions
    X_train_pred = simulate(models, sign_mask, **data["train_dataset"].get_simulation_data())
    X_test_pred = simulate(models, sign_mask, **data["test_dataset"].get_simulation_data())
    
    # Evaluate predictions
    evaluate_predictions(data["X_train"], X_train_pred, data["X_columns"], "Train")
    evaluate_predictions(data["X_test"], X_test_pred, data["X_columns"], "Test")
    
    # Visualize results
    select_runs = config["hybrid_model_params"]["visualization"]["select_runs"]

    plot_predicted_profile(
        data["X_test"].transpose(1, 2, 0),
        X_test_pred.transpose(1, 2, 0),
        data["X_columns"],
        select_runs=select_runs,
    )


def test_pipeline(config):
    """Execute testing pipeline"""
    # Load only test dataset
    data = load_datasets(config, mode="test")
    
    # Load saved models
    models, sign_mask = load_saved_models(config)
    
    # Generate predictions for test set
    X_test_pred = simulate(models, sign_mask, **data["test_dataset"].get_simulation_data())
    
    # Evaluate predictions
    evaluate_predictions(data["X_test"], X_test_pred, data["X_columns"], "Test")
    
    # Visualize results
    select_runs = config["hybrid_model_params"]["visualization"]["select_runs"]
    plot_predicted_profile(
        data["X_test"].transpose(1, 2, 0),
        X_test_pred.transpose(1, 2, 0),
        data["X_columns"],
        select_runs=select_runs,
    )


def predict_pipeline(config):
    """Execute prediction pipeline"""
    # Load prediction dataset
    data = load_datasets(config, mode="predict")
    
    # Load saved models
    models, sign_mask = load_saved_models(config)
    
    # Generate predictions
    X_pred = simulate(models, sign_mask, **data["predict_dataset"].get_simulation_data())
    
    # Visualize predictions (without true values)
    plot_predicted_profile(
        None,
        X_pred.transpose(1, 2, 0),
        data["X_columns"],
        predict_mode=True
    )
    
    # Save predictions if needed
    if config['hybrid_model_params'].get('save_predictions', True):
        save_dir = os.path.join(config['results_dir'], 'predictions')
        os.makedirs(save_dir, exist_ok=True)
        data['predict_dataset'].save_predictions(X_pred, save_dir)
        print(f"\nPredictions saved to {save_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train, test, or predict with hybrid bioreactor model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "predict"],
        default="train",
        help="Mode to run the script in: train (full training pipeline), test (evaluation only), or predict (generate predictions)",
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Get configuration
    config = get_default_config()
    config["hybrid_model_params"]["mode"] = args.mode

    # Execute pipeline based on mode
    if args.mode == "train":
        print("Executing training pipeline...")
        train_pipeline(config)
    elif args.mode == "test":
        print("Executing testing pipeline...")
        test_pipeline(config)
    else:
        print("Executing prediction pipeline...")
        predict_pipeline(config)


if __name__ == "__main__":
    main()
