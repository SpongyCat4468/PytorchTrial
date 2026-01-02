# PYTORCH IMPLEMENTATION OF LINEAR REGRESSION USING nn.Module, nn.MSELoss, AND optim.Adam (Professional)
import torch.nn as nn
import torch.optim as optim
import torch

learning_rate = 0.01
X = torch.randn(10, 1)
true_W = torch.tensor([[2.0]])
true_b = torch.tensor([1.0])
Y = X @ true_W + true_b # + 0.1 * torch.randn(N, D_out) (noise)

class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear_layer(x)
model = LinearRegressionModel(1, 1)

optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    
loss_fn = nn.MSELoss()

print("Starting training...")

epochs = 1000
for epoch in range(epochs):
    y_hat = model(X)
    loss = loss_fn(y_hat, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        W, b = model.linear_layer.weight.item(), model.linear_layer.bias.item()
        print(f'Epoch {epoch + 1}: loss = {loss.item()}, W = {W:.3f}, b = {b:.3f}')