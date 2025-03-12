import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size=3, action_size=1, hidden_size=(400, 300), seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (tuple): Tuple containing size of hidden layers
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Create a ModuleList to hold all linear layers
        self.layers = nn.ModuleList()

        # Input layer
        in_size = state_size

        # Create hidden layers dynamically based on hidden_size tuple
        for h_size in hidden_size:
            self.layers.append(nn.Linear(in_size, h_size))
            in_size = h_size

        # Output layer
        self.layers.append(nn.Linear(in_size, action_size))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize layer parameters."""
        # Initialize all hidden layers with kaiming uniform
        for i in range(len(self.layers) - 1):
            nn.init.kaiming_uniform_(
                self.layers[i].weight, mode="fan_in", nonlinearity="relu"
            )

        # Initialize the final layer with uniform distribution
        nn.init.uniform_(self.layers[-1].weight, -3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        # Apply ReLU to all hidden layers
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))

        # Apply tanh to the output layer
        return F.tanh(self.layers[-1](x))
    

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size=3, action_size=1, hidden_size=(400, 300), seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (tuple): Tuple containing size of hidden layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # First layer processes state only
        self.fc1 = nn.Linear(state_size, hidden_size[0])
        
        # Second layer processes concatenated state features and action
        self.fc2 = nn.Linear(hidden_size[0] + action_size, hidden_size[1])
        
        # Output layer produces a single Q-value
        self.fc3 = nn.Linear(hidden_size[1], 1)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize layer parameters using the same approach as Actor."""
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', 
                                nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', 
                                nonlinearity='relu')
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)