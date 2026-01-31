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

