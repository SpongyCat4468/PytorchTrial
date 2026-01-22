import torch.nn as nn
import torch
from pathlib import Path
MODEL_PATH = Path("models")

MODEL_NAME = "01_pytorch_basic.pt" 
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


# Loading a pytorch model
class LineRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.noise = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (x**2) + self.noise * x + self.bias
    
model = LineRegressionModel()
model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, weights_only=True))
print(model.state_dict())