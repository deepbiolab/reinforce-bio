import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
pio.templates.default = "plotly_white"
pcolors = px.colors.qualitative.T10
pcolors25 = px.colors.qualitative.Alphabet


def plot_predicted_profile(X, X_pred, X_columns, select_runs=[0], height=1000):
    max_cols_per_row = 5
    num_columns = len(X_columns)
    num_rows = (num_columns + max_cols_per_row) // max_cols_per_row

    fig = make_subplots(
        rows=num_rows, cols=min(num_columns, max_cols_per_row), subplot_titles=X_columns
    )

    color_palette = px.colors.qualitative.Plotly

    for idx, j in enumerate(select_runs):
        color = color_palette[idx % len(color_palette)]
        for i, c in enumerate(X_columns):
            row = i // max_cols_per_row + 1
            col = i % max_cols_per_row + 1
            show_legend = i == 0
            fig.add_trace(
                go.Scatter(
                    x=list(range(15)),
                    y=X[:, i, j],
                    name=f"Run {j} Observed",
                    marker=dict(color=color),
                    showlegend=show_legend,
                    legendgroup=f"group_{j}",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(15)),
                    y=X_pred[:, i, j],
                    name=f"Run {j} Predicted",
                    line=dict(dash="dash"),
                    marker=dict(color=color),
                    showlegend=show_legend,
                    legendgroup=f"group_{j}",
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        showlegend=True,
        title_text="Process variable evolution for selected runs",
        height=height,
    )
    fig.show()
    

def plot_sensitivity_results(total_results, time_points, X_columns):
    """Plot sensitivity analysis results in a single row with four columns"""
    for param_name, sensitivity_results in total_results.items():
        # Create figure with a single row and four columns
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'Sensitivity Analysis for {param_name}', fontsize=16, y=1.05)
        
        # Get parameter values
        param_values = [res['param_value'] for res in sensitivity_results]
        
        # Create color map
        norm = plt.Normalize(min(param_values), max(param_values))
        cmap = plt.cm.viridis
        
        var_labels = ['VCD (10^6 cells/mL)', 'Glucose (g/L)', 
                    'Lactate (g/L)', 'Product Titer (mg/L)']
        
        for i, res in enumerate(sensitivity_results):
            predictions = res['predictions']
            param_value = res['param_value']
            color = cmap(norm(param_value))
            
            # Plot each state variable in its respective column
            for j in range(4):
                axes[j].plot(time_points, predictions[:, j], '-',
                            label=f"{param_name}={param_value:.2f}",
                            color=color)
                
                axes[j].set_title(X_columns[j])
                axes[j].set_xlabel('Time (days)')
                axes[j].set_ylabel(var_labels[j])
                axes[j].grid(True, alpha=0.3)
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
        fig.colorbar(sm, cax=cbar_ax, label=param_name)
        
        plt.subplots_adjust(right=0.95)  # Adjust to make room for colorbar
        plt.show()
        

def plot_optimal_conditions(best_result, X_columns):
    """Plot the predicted profiles for optimal conditions"""
    predictions = best_result['predictions']
    time_points = best_result['time_points']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Predicted Profiles for Optimal Process Conditions', fontsize=16)
    
    var_labels = ['VCD (10^6 cells/mL)', 'Glucose (g/L)',
                 'Lactate (g/L)', 'Product Titer (mg/L)']
    
    for i in range(4):
        axes[i].plot(time_points, predictions[:, i], 'o-')
        axes[i].set_title(X_columns[i])
        axes[i].set_xlabel('Time (days)')
        axes[i].set_ylabel(var_labels[i])
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()
    

def plot_validation_comparison(optimal_result, X_train, X_test, X_columns, time_points):
    """
    Visualize the comparison between optimal results and historical data.
    
    Args:
        optimal_result: Dictionary containing the optimization results
        X_train: Training data array (n_samples, n_timesteps, n_features)
        X_test: Test data array (n_samples, n_timesteps, n_features)
        X_columns: List of column names for the features
        time_points: Array of time points for x-axis
    """
    # Create comparison plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Optimal Results vs Training/Test Data Distribution', fontsize=16, y=1.05)
    
    var_labels = ['VCD (10^6 cells/mL)', 'Glucose (g/L)',
                 'Lactate (g/L)', 'Product Titer (mg/L)']
    
    # Create custom color palette
    train_color = 'lightgray'
    test_color = 'lightblue'
    optimal_color = 'red'
    
    for i in range(4):
        # Plot training data with transparency
        for j in range(len(X_train)):
            axes[i].plot(time_points, X_train[j, :, i], 
                        color=train_color, alpha=0.2, 
                        linewidth=1)
        
        # Plot test data with transparency
        for j in range(len(X_test)):
            axes[i].plot(time_points, X_test[j, :, i], 
                        color=test_color, alpha=0.2,
                        linewidth=1)
        
        # Calculate and plot mean curves
        train_mean = np.mean(X_train[:, :, i], axis=0)
        test_mean = np.mean(X_test[:, :, i], axis=0)
        
        axes[i].plot(time_points, train_mean, '--', 
                    color='gray', linewidth=2, 
                    label='Train Mean')
        axes[i].plot(time_points, test_mean, '--', 
                    color='blue', linewidth=2, 
                    label='Test Mean')
        
        # Plot optimal prediction
        axes[i].plot(time_points, optimal_result['predictions'][:, i], 
                    color=optimal_color, linewidth=2.5, 
                    label='Optimal')
        
        # Customize plot appearance
        axes[i].set_title(X_columns[i], fontsize=12, pad=10)
        axes[i].set_xlabel('Time (days)', fontsize=10)
        axes[i].set_ylabel(var_labels[i], fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(labelsize=9)
        
        # Add legend only to the first plot
        if i == 0:
            axes[i].legend(fontsize=9)
        
        # Set y-axis limits with some padding
        data_max = max(np.max(X_train[:, :, i]), 
                      np.max(X_test[:, :, i]),
                      np.max(optimal_result['predictions'][:, i]))
        data_min = min(np.min(X_train[:, :, i]), 
                      np.min(X_test[:, :, i]),
                      np.min(optimal_result['predictions'][:, i]))
        padding = (data_max - data_min) * 0.1
        axes[i].set_ylim(data_min - padding, data_max + padding)
    
    plt.show()

def create_optimization_animation(optimization_results, time_points, param_ranges, output_path='optimization_progress.gif'):
    """
    Create an animation showing the optimization progress with parameter indicators.
    
    Args:
        optimization_results: List of results from optimize_process_conditions
        time_points: Array of time points used in simulation
        param_ranges: Dictionary of parameter ranges
        output_path: Path to save the animation GIF
    """
    
    # Setup the figure with GridSpec
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, width_ratios=[2, 1])
    
    # Create axes
    ax1 = fig.add_subplot(gs[0, 0])  # Titer profile
    ax2 = fig.add_subplot(gs[1, 0])  # Optimization progress
    ax3 = fig.add_subplot(gs[:, 1])  # Parameter indicators
    
    # Setup the first subplot - Titer profile
    line1, = ax1.plot([], [], 'b-', alpha=0.5, label='Current Best')
    scatter1 = ax1.scatter([], [], c='r', alpha=0.5, s=50)
    ax1.set_xlim(0, len(time_points))
    ax1.set_ylim(0, max([r['final_titer'] for r in optimization_results]) * 1.1)
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Titer (mg/L)')
    ax1.grid(True)
    ax1.legend()
    
    # Setup the second subplot - Optimization progress
    line2, = ax2.plot([], [], 'k-')
    ax2.set_xlim(0, len(optimization_results))
    ax2.set_ylim(500, max([r['final_titer'] for r in optimization_results]) * 1.1)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Titer (mg/L)')
    ax2.grid(True)
    
    # Setup the third subplot - Parameter indicators
    param_names = list(param_ranges.keys())
    num_params = len(param_names)
    bar_positions = np.arange(num_params)
    cmap = plt.cm.Set3
    colors = [cmap(i/num_params) for i in range(num_params)]
    bars = ax3.bar(bar_positions, [0] * num_params, color=colors)

    
    # Normalize parameter ranges to 0-1 scale
    param_normalizers = {}
    for param, (min_val, max_val) in param_ranges.items():
        param_normalizers[param] = lambda x, min_val=min_val, max_val=max_val: (x - min_val) / (max_val - min_val)
    
    ax3.set_ylim(0, 1)
    ax3.set_xticks(bar_positions)
    ax3.set_xticklabels(param_names, rotation=45)
    ax3.set_ylabel('Normalized Parameter Value')
    ax3.grid(True, axis='y')
    
    # Prepare animation data
    best_so_far = None
    best_titers = []
    anim_data = []
    
    for result in optimization_results:
        if best_so_far is None or result['final_titer'] > best_so_far['final_titer']:
            best_so_far = result
        
        best_titers.append(best_so_far['final_titer'])
        
        # Normalize current parameters
        normalized_params = []
        for param in param_names:
            norm_value = param_normalizers[param](best_so_far['params'][param])
            normalized_params.append(norm_value)
        
        anim_data.append({
            'time_points': time_points,
            'predictions': best_so_far['predictions'][:, 3],  # Titer predictions
            'final_titer': best_so_far['final_titer'],
            'best_titers': best_titers.copy(),
            'normalized_params': normalized_params
        })
    
    def update(frame):
        # Update titer profile plot
        line1.set_data(anim_data[frame]['time_points'], 
                      anim_data[frame]['predictions'])
        scatter1.set_offsets([[time_points[-1], 
                             anim_data[frame]['final_titer']]])
        
        # Update optimization progress plot
        line2.set_data(range(len(anim_data[frame]['best_titers'])), 
                      anim_data[frame]['best_titers'])
        
        # Update parameter indicators
        for bar, height in zip(bars, anim_data[frame]['normalized_params']):
            bar.set_height(height)
        
        return [line1, scatter1, line2] + list(bars)
    
    # Create and save animation
    anim = FuncAnimation(fig, update, frames=len(anim_data), 
                        interval=100, blit=True)
    writer = PillowWriter(fps=5)
    anim.save(output_path, writer=writer, dpi=300)
    plt.close()

