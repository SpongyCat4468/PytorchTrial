import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Data - More points for better coverage
a, b, c, d, e = -3.5, -12.2, 3.4, -9.1, 6.7
start, end, step = 0, 10000, 50
X = torch.arange(start, end, step, dtype=torch.float32).unsqueeze(dim=1).to(device)
Y = (a * (X ** 4) + b * (X ** 3) + c * (X ** 2) + d * X + e).to(device)

X_mean, X_std = X.mean(), X.std()
Y_mean, Y_std = Y.mean(), Y.std()

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

# CRITICAL FIX: Random splitting instead of sequential
# Sequential splitting causes the test set to only see extreme polynomial values
torch.manual_seed(42)
indices = torch.randperm(len(X))
train_split = int(0.80 * len(X))
train_idx = indices[:train_split]
test_idx = indices[train_split:]

X_train, Y_train = X_norm[train_idx], Y_norm[train_idx]
X_test, Y_test = X_norm[test_idx], Y_norm[test_idx]

# Sort for better visualization
train_sort_idx = torch.argsort(X_train.squeeze())
test_sort_idx = torch.argsort(X_test.squeeze())
X_train_sorted = X_train[train_sort_idx]
Y_train_sorted = Y_train[train_sort_idx]
X_test_sorted = X_test[test_sort_idx]
Y_test_sorted = Y_test[test_sort_idx]

# Plot predictions function
def plot_predictions(train_data=X_train_sorted,
                     train_labels=Y_train_sorted,
                     test_data=X_test_sorted,
                     test_labels=Y_test_sorted,
                     predictions=None):
    plt.figure(figsize=(12, 7))
    plt.scatter(train_data.cpu(), train_labels.cpu(), c="b", s=4, alpha=0.6, label="Training data")
    plt.scatter(test_data.cpu(), test_labels.cpu(), c="g", s=4, alpha=0.6, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data.cpu(), predictions.cpu(), c="r", s=4, alpha=0.8, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.title("Polynomial Regression with Random Train/Test Split")
    plt.xlabel("X (normalized)")
    plt.ylabel("Y (normalized)")
    plt.grid(True, alpha=0.3)
    plt.show()

# Model architecture
class PolynomialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),  # Reduced dropout slightly
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

torch.manual_seed(42)
model = PolynomialModel().to(device)

print(f"Running model on device: {device}")
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n")

best_test_loss = float('inf')
best_model_state = None
best_epoch = 0
patience = 800
patience_counter = 0

epochs, lr = 8000, 0.001
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=150
)

for epoch in range(epochs):
    # Training
    model.train()
    Y_pred = model(X_train)
    train_loss = loss_fn(Y_pred, Y_train)
    
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, Y_test)

    scheduler.step(test_loss)

    # Save best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss.item()
        best_epoch = epoch + 1
        patience_counter = 0
        best_model_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    # Progress logging
    if (epoch + 1) % 500 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch + 1:4d} | Train: {train_loss:.6f} | Test: {test_loss:.6f} | LR: {current_lr:.2e}")

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\n{'='*60}")
    print(f"Loaded best model from epoch: {best_epoch}")
    print(f"Best Test Loss: {best_test_loss:.6f}")
else:
    print("Warning: No best model state was saved!")

# Final evaluation
model.eval()
with torch.inference_mode():
    train_pred = model(X_train)
    test_pred_sorted = model(X_test_sorted)
    
    final_train_loss = loss_fn(train_pred, Y_train)
    final_test_loss = loss_fn(test_pred_sorted, Y_test_sorted)
    
    # Calculate R² score
    ss_res_train = torch.sum((Y_train - train_pred) ** 2)
    ss_tot_train = torch.sum((Y_train - Y_train.mean()) ** 2)
    r2_train = 1 - (ss_res_train / ss_tot_train)
    
    ss_res_test = torch.sum((Y_test_sorted - test_pred_sorted) ** 2)
    ss_tot_test = torch.sum((Y_test_sorted - Y_test_sorted.mean()) ** 2)
    r2_test = 1 - (ss_res_test / ss_tot_test)
    
    print(f"{'='*60}")
    print(f"Final Train Loss: {final_train_loss:.6f} | R²: {r2_train:.4f}")
    print(f"Final Test Loss:  {final_test_loss:.6f} | R²: {r2_test:.4f}")
    print(f"Train/Test Ratio: {final_train_loss/final_test_loss:.3f}")
    print(f"{'='*60}\n")
    
    plot_predictions(predictions=test_pred_sorted)

from pathlib import Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "02_polynomial.pt" #common file type -> .pt / .pth
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

if (input("save (y/n)").lower() == "y"):
    torch.save(model.state_dict(), MODEL_SAVE_PATH)