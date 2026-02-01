# Make classification data
# import pandas as pd
# import matplotlib.pyplot as plt
# import sklearn
from sklearn.datasets import make_circles
import torch
import torch.nn as nn

n_samples = 1000

# Create circle
X, y = make_circles(n_samples, noise=0.03, random_state=42) # random_state -> setting seed  
'''
X
[0.75424625  0.23148074]
[-0.75615888  0.15325888]
[-0.81539193  0.17328203]
[-0.39373073  0.69288277]
[ 0.44220765 -0.89672343]

y
[1 1 1 1 0]


circles = pd.DataFrame({"X": X[:, 0],
                        "Y": X[:, 1],
                        "label": y})
print(circles.head(10))

plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
plt.show()
'''
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Splitting data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 0.2 -> 20%

# Buidling a model to classify the blue and red dots (outer and inner circle)
device = "cuda" if torch.cuda.is_available else "cpu"

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layer(x)

model = CircleModel().to(device)


predictions = model(X_test.to(device))

'''
print(predictions[:10])
print(y_test[:10])
>-------<
tensor([[0.0388],
        [0.0241],
        [0.0511],
        [0.0387],
        [0.3365],
        [0.3404],
        [0.1965],
        [0.2550],
        [0.0446],
        [0.0234]], device='cuda:0', grad_fn=<SliceBackward0>)
tensor([1., 0., 1., 0., 1., 1., 0., 0., 1., 0.])
'''
# For regression: MAE or MSE loss (L1Loss or MSELoss)
# For classification: binary cross entropy loss (BCELoss) or categorical cross entropy 
loss_fn = nn.BCEWithLogitsLoss()

# Common optimizer: Adam, SGD (can try others)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

# Calculate accuracy - out of 100 examples, what percentage does the model succeed to predict?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

model.eval()
with torch.inference_mode():
    y_logits = model(X_test.to(device))[:5]

y_pred_prob = torch.sigmoid(y_logits)
y_pred = torch.round(y_pred_prob)
# print(f"Predictions: {list(item.item() for item in y_pred)}\nReal: {list(item.item() for item in y_test[:5])}")
y_pred_labels = torch.round(torch.sigmoid(model(X_test.to(device))[:5]))
# print(torch.eq(y_pred.squeeze(), y_pred_labels.squeeze()))
'''
Predictions: [0.0, 1.0, 0.0, 1.0, 0.0]
Real: [1.0, 0.0, 1.0, 0.0, 1.0]
tensor([True, True, True, True, True], device='cuda:0')
'''

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    # Training
    model.train()
    # Turning logits into pred_prob and then into pred
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    # Calculate the accuracy / loss
    loss = loss_fn(y_logits, y_train)
    accuracy = accuracy_fn(y_train, y_pred)
    # Optimizer zero grad
    optimizer.zero_grad()
    # Loss backward
    loss.backward()
    # Optimizer step
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)

    # Print out result for 100 epochs
    if (epochs + 1) % 100 == 0:
        print(f"Epoch {epochs + 1} | Loss: {test_loss:.5f} | Accuracy: {test_acc:.5f} | Test Loss: {test_loss:.2f}%")
