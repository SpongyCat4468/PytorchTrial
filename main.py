""" FROM SCRATCH IMPLEMENTATION OF LINEAR REGRESSION USING PYTORCH
N = 10
D_in = 1
D_out = 1

X = torch.randn(N, D_in)
true_W = torch.tensor([[2.0]])
true_b = torch.tensor([1.0])
Y = X @ true_W + true_b # + 0.1 * torch.randn(N, D_out) (noise)

learning_rate, epochs = 0.01, 1001

W, b = torch.randn(1, 1, requires_grad=True), torch.randn(1, requires_grad=True)

for epoch in range(epochs):
    y_hat = X @ W + b
    loss = torch.mean((Y - y_hat) ** 2)

    loss.backward()

    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: loss = {loss.item()}, W = {W.item():.3f}, b = {b.item():.3f}')

    W.grad.zero_()
    b.grad.zero_()
    

print(f'Final parameters: W = {W.item():.3f}, b = {b.item():.3f}')
print(f'True parameters: W = {true_W.item():.3f}, b = {true_b.item():.3f}')
"""

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

epochs = 420
for epoch in range(epochs):
    y_hat = model(X)
    loss = loss_fn(y_hat, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        W, b = model.linear_layer.weight.item(), model.linear_layer.bias.item()
        print(f'Epoch {epoch + 1}: loss = {loss.item()}, W = {W:.3f}, b = {b:.3f}')