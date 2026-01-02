import torch.nn as nn
import torch.optim as optim
import torch

learning_rate = 0.01

# Your data
angles = [0, 0.174532925, 0.34906585, 0.523598776, 0.698131701, 0.872664626, 1.047197551, 1.221730476, 1.396263402, 1.570796327]
voltages = [5.2264, 5.1709, 5.0379, 5.0147, 4.7384, 4.071, 3.619, 2.679, 1.3579, 0.3957]

# Convert to tensors and reshape
X = torch.tensor(angles, dtype=torch.float32).reshape(-1, 1)
Y = torch.tensor(voltages, dtype=torch.float32).reshape(-1, 1)

class VoltageAngleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # We'll use a linear layer on cos(angle)
        self.linear_layer = nn.Linear(1, 1)

    def forward(self, x):
        # Apply cosine transformation before linear layer
        cos_x = torch.cos(x)
        return self.linear_layer(cos_x)

model = VoltageAngleModel()

optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
loss_fn = nn.MSELoss()

print("Starting training...")
print(f"Target formula: voltage = a * cos(angle) + b\n")

epochs = 10000
for epoch in range(epochs):
    y_hat = model(X)
    loss = loss_fn(y_hat, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        a = model.linear_layer.weight.item()
        b = model.linear_layer.bias.item()
        print(f'Epoch {epoch + 1}: loss = {loss.item():.6f}, a = {a:.3f}, b = {b:.3f}')

# Final results
a = model.linear_layer.weight.item()
b = model.linear_layer.bias.item()
print(f'\nFinal parameters: a = {a:.3f}, b = {b:.3f}')
print(f'Formula: voltage = {a:.3f} * cos(angle) + {b:.3f}')

print("\nTesting predictions:")
for angle, true_voltage in zip(angles[:3], voltages[:3]):
    predicted = model(torch.tensor([[angle]])).item()
    print(f'Angle: {angle:.3f} rad, True: {true_voltage:.3f}V, Predicted: {predicted:.3f}V')

# 1
# Epoch 1000: loss = 0.187861, a = 3.846, b = 1.399
# Epoch 10000: loss = 0.088929, a = 4.761, b = 0.772
# Epoch 100000: same result as no.10000
