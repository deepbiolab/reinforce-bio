# Reinforce-Bio: Leveraging Reinforcement Learning and Hybrid Modeling for Bioprocess Optimization
---

![model-arch](./assets/reinforce-bio.svg)

Reinforce-Bio is a research project focused on leveraging advanced reinforcement learning to optimize bioprocess parameters and improve efficiency in biological systems. 

## Overview

Bioprocess optimization is a critical area in biotechnology, aiming to enhance the production of biologics such as monoclonal antibodies, vaccines, and enzymes. The project explores hybrid modeling, combining mechanistic models and machine learning, to better understand and optimize complex biological systems.

![](./assets/optimization_progress.gif)

Currently, this repository contains models and methods for hybrid modeling and simulation of bioprocesses, with an emphasis on integrating data-driven approaches like Gaussian Process (GP) modeling. In future, the project aims to develop a reinforcement learning optimization model that can identify the optimal process conditions for maximizing product yield and minimizing costs.

### Key Features

- **Hybrid Gaussian Process Models**: Combines mechanistic and data-driven approaches to model bioprocess dynamics.
- **Simulation Module**: Predict bioprocess states under varying process parameters.
- **Sensitivity Analysis**: Evaluate the impact of key process parameters on bioprocess outcomes.
- **Reinforcement Learning Optimization Module**: Identify optimal process conditions for maximizing product yield and minimizing costs.

## Simulation: Hybrid Model

The hybrid model integrates both mechanistic and machine learning components to capture the dynamics of a bioprocess. It uses Gaussian Process (GP) regression to model the derivatives of key state variables (e.g., cell density, glucose concentration) and employs Ordinary Differential Equations (ODEs) to simulate state evolution over time.

### Key Data Scopes

1. **State Variables**:
   - Cell density (VCD)
   - Glucose concentration
   - Lactate concentration
   - Product titer

2. **Process Conditions**:
   - Feed rates
   - Feed Start Day
   - Feed End Day
   - Initial conditions (e.g., initial glucose and cell density)


## Optimization: Deep Deterministic Policy Gradient (DDPG)

The optimization module leverages the hybrid model to identify optimal process conditions. By simulating a range of parameter combinations, the framework evaluates their impact on product yield and other performance metrics, ultimately selecting the best conditions.

## Future Work

The next phase of this project will focus on integrating reinforcement learning (RL) to design adaptive feeding strategies that dynamically adjust to bioprocess conditions. This will further enhance the flexibility and robustness of bioprocess optimization.

## Getting Started

### Prerequisites

- Python 3.10.x

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/deepbiolab/reinforce-bio.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


> For more details, refer to the documentation in the `report/` folder or the notebooks in the `notebook/` folder.


## Citation

If you find this project useful in your research, please cite it as follows:
```
@software{reinforce_bio2025,
  author       = {Tim-Lin},
  title        = {Reinforce-Bio: Leveraging Reinforcement Learning and Hybrid Modeling for Bioprocess Optimization},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/deepbiolab/reinforce-bio}
}
```