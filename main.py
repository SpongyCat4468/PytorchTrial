import torch.nn as nn
import torch.optim as optim
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

time1 = time.time()

learning_rate = 0.01

# Move tensors to device
X = torch.randn(10, 1).to(device)
true_W = torch.tensor([[2.0]]).to(device)
true_b = torch.tensor([1.0]).to(device)
Y = X @ true_W + true_b

class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear_layer(x)

# Move model to device
model = LinearRegressionModel(1, 1).to(device)

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

print(f"Training complete. Completed in {time.time() - time1:.2f} seconds.")