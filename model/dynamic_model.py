from base.base_model import BaseModel
import torch.nn as nn

# Creating a configurable dense model
class DenseModel(BaseModel):
    def __init__(self, hidden_layers=1, neurons_per_layer=1, activation_hidden='relu', activation_output='linear'):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output

        self.layer = nn.ModuleList()

        # Creating activation map
        activation_map = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'linear': nn.Identity()
        }

        # Creating hidden layers
        # If hidden_layers is 0, then the model is a linear model
        if hidden_layers == 0:
            self.layer.append(nn.Linear(1, 1))
        else:
            self.layer.append(nn.Linear(1, neurons_per_layer))
            if activation_hidden != 'linear':
                self.layer.append(activation_map[activation_hidden])
            for _ in range(hidden_layers - 1):
                self.layer.append(nn.Linear(neurons_per_layer, neurons_per_layer))
                if activation_hidden != 'linear':
                    self.layer.append(activation_map[activation_hidden])
            self.layer.append(nn.Linear(neurons_per_layer, 1))
        
        # Output Activation
        if activation_output != 'linear':
            self.layer.append(activation_map[activation_output])


    def forward(self, x):
        # Reshape the input if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
            
        for layer in self.layer:
            x = layer(x)
        return x