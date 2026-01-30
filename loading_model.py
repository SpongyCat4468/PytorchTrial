import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt

MODEL_PATH = Path("models")
MODEL_NAME = "01_polynomial.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Original polynomial coefficients (same as training)
a, b, c, d, e = -3.5, -12.2, 3.4, -9.1, 6.7

def precise(x: int):
    return a * (x**4) + b * (x**3) + c * (x**2) + d * x + e
# GENERATE COMPLETELY NEW DATA
# Different range and step size to prove the model generalizes
print("Generating completely NEW data (different from training)...")
start, end, step = 1000, 9000, 75  # Different range and step!
X_new = torch.arange(start, end, step, dtype=torch.float32).unsqueeze(dim=1)
Y_new = (a * (X_new ** 4) + b * (X_new ** 3) + c * (X_new ** 2) + d * X_new + e)

print(f"New data: {len(X_new)} points from X={start} to X={end-step}")

# Use the SAME normalization parameters as training
# (In practice, you'd save these with the model, but we can recalculate since we know the formula)
X_mean_original = 5000.0  # Approximate mean of [0, 10000, step=50]
X_std_original = 2886.75  # Approximate std
Y_mean_original = -1.7502e+15  # You'd normally save these
Y_std_original = 4.9087e+15

# Better approach: recalculate from the full range
X_full_range = torch.arange(0, 10000, 50, dtype=torch.float32).unsqueeze(dim=1)
Y_full_range = (a * (X_full_range ** 4) + b * (X_full_range ** 3) + 
                c * (X_full_range ** 2) + d * X_full_range + e)
X_mean = X_full_range.mean()
X_std = X_full_range.std()
Y_mean = Y_full_range.mean()
Y_std = Y_full_range.std()

print(f"Normalization stats - X: mean={X_mean:.2f}, std={X_std:.2f}")
print(f"Normalization stats - Y: mean={Y_mean:.2e}, std={Y_std:.2e}\n")

# Normalize the new data
X_new_norm = (X_new - X_mean) / X_std
Y_new_norm = (Y_new - Y_mean) / Y_std

# Define the model architecture (must match training)
class PolynomialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# Load the saved model
print(f"Loading model from {MODEL_SAVE_PATH}...")
model = PolynomialModel()
model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, weights_only=True))
model.eval()
print("Model loaded successfully!\n")

# Make predictions on the NEW data
with torch.inference_mode():
    Y_pred_norm = model(X_new_norm)

# Calculate loss and R²
loss_fn = nn.MSELoss()
loss = loss_fn(Y_pred_norm, Y_new_norm)

ss_res = torch.sum((Y_new_norm - Y_pred_norm) ** 2)
ss_tot = torch.sum((Y_new_norm - Y_new_norm.mean()) ** 2)
r2_score = 1 - (ss_res / ss_tot)

print("="*60)
print("RESULTS ON COMPLETELY NEW DATA:")
print(f"Test Loss (MSE): {loss:.6f}")
print(f"R² Score: {r2_score:.6f}")
print("="*60)

# Visualization
plt.figure(figsize=(14, 6))

# Plot 1: Normalized space (what the model sees)
plt.subplot(1, 2, 1)
plt.scatter(X_new_norm.cpu(), Y_new_norm.cpu(), c="blue", s=20, alpha=0.6, label="True values")
plt.scatter(X_new_norm.cpu(), Y_pred_norm.cpu(), c="red", s=20, alpha=0.6, label="Predictions")
plt.xlabel("X (normalized)")
plt.ylabel("Y (normalized)")
plt.title("Normalized Space - Model Predictions vs True")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Original space (easier to interpret)
Y_pred_original = Y_pred_norm * Y_std + Y_mean
plt.subplot(1, 2, 2)
plt.scatter(X_new.cpu(), Y_new.cpu(), c="blue", s=20, alpha=0.6, label="True values")
plt.scatter(X_new.cpu(), Y_pred_original.cpu(), c="red", s=20, alpha=0.6, label="Predictions")
plt.xlabel("X (original)")
plt.ylabel("Y (original)")
plt.title("Original Space - Model Predictions vs True")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional test: Sample specific points
print("\nSample predictions at specific X values:")
test_points = torch.tensor([[2000.0], [5000.0], [8000.0]])
test_points_norm = (test_points - X_mean) / X_std

with torch.inference_mode():
    test_preds_norm = model(test_points_norm)
    test_preds = test_preds_norm * Y_std + Y_mean

for i, x_val in enumerate(test_points.squeeze()):
    y_true = (a * (x_val ** 4) + b * (x_val ** 3) + c * (x_val ** 2) + d * x_val + e)
    y_pred = test_preds[i].item()
    error = abs(y_true - y_pred)
    error_pct = (error / abs(y_true)) * 100
    print(f"X={x_val:.0f}: Truae={y_true:.2e}, Pred={y_pred:.2e}, Error={error_pct:.3f}%")