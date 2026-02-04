import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 
from pathlib import Path

from tqdm.auto import tqdm
from torchvision import datasets
from torchvision.transforms import ToTensor
from helper_functions import accuracy_fn

from timeit import default_timer as timer

# Preparing data
device = "cpu" if torch.cuda.is_available else "cpu"
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor(), target_transform=None)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor(), target_transform=None)
'''
{'T-shirt/top': 0, 'Trouser': 1, 'Pullover': 2, 'Dress': 3, 'Coat': 4, 'Sandal': 5, 'Shirt': 6, 'Sneaker': 7, 'Bag': 8, 'Ankle boot': 9}

print(image.shape, label)
torch.Size([1, 28, 28]), 9
            |-> (color channels, height, width)
1 color channel -> grey scale image

Visualizing our images
'''
'''
image, label = train_data[0]
class_items = {value: key for key, value in train_data.class_to_idx.items()}

plt.title(class_items[label])
plt.axis(False)
plt.imshow(image.squeeze(), cmap="gray")
plt.show()
'''

'''
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_items[label])
    plt.axis(False)
plt.show()
'''

class_items = {value: key for key, value in train_data.class_to_idx.items()}
# Preparing dataloader (data loader turns our data into a python iterable) -> breaking data into small batches so memory can load it in
BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, BATCH_SIZE, shuffle=False) # dont shuffle -> easier to evaluate the model


'''
torch.manual_seed(67)
train_features_batch, train_label_batch = next(iter(train_dataloader))
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_label_batch[random_idx].item()
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_items[label])
plt.axis(False)
plt.show()
'''
 
# Building a computer vision model
# nn.Flatten() -> dim [color_channels, height, width] -> [color_channels, height*width]
class ComputerVisonModel(nn.Module):
    def __init__(self, input, output, hidden):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output)
        )

    def forward(self, x):
        return self.layer(x)
    
def print_train_time(start: float, end: float) -> float:
    total_time = end - start
    print(f"Training time: {total_time:.3f} seconds")
    return total_time
torch.manual_seed(67)
model = ComputerVisonModel(784, 10, 32).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 10
torch.manual_seed(67)
train_time_start = timer()
for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch + 1}\n------")

    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader): # X -> image, y -> label
        model.train()
        y_pred = model(X.to(device))

        loss = loss_fn(y_pred, y.to(device))
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")
    train_loss /= len(train_dataloader)

    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            # test_pred in format of logits
            test_pred = model(X_test.to(device))
            test_loss += loss_fn(test_pred, y_test.to(device))
            test_acc += accuracy_fn(y_test.to(device), test_pred.argmax(dim=1))
            
        # Calculate test loss average per batch
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    if epoch == 0:
        print(f"\nTrain Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    else:
        better_train_loss = True if (train_loss.item() < prev_train_loss) else False
        better_test_loss = True if (test_loss.item() < prev_test_loss) else False
        better_acc = True if test_acc > prev_acc else False
        pointer_train_loss = "↑" if better_train_loss else "↓"
        pointer_test_loss = "↑" if better_test_loss else "↓"
        pointer_acc = "↑" if better_acc else "↓"
        print((train_loss.item() < prev_train_loss), (test_loss.item() < prev_test_loss), better_train_loss, better_test_loss)
        print(f"\nTrain Loss: {train_loss:.4f} {pointer_train_loss}| Test Loss: {test_loss:.4f} {pointer_test_loss}| Test Acc: {test_acc:.2f}% {pointer_acc}")
        
    prev_train_loss, prev_test_loss, prev_acc = train_loss.item(), test_loss.item(), test_acc
train_time_end = timer()
total_train_time = print_train_time(train_time_start, train_time_end)

torch.manual_seed(42)
def eval_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, device: str, accuracy_fn):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X.to(device))

            loss += loss_fn(y_pred, y.to(device))
            acc += accuracy_fn(y.to(device), y_pred.argmax(dim=1))
        
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, 
            "model_loss": loss.item(),
            "model_acc": acc}

model_results = eval_model(model, test_dataloader, loss_fn, device, accuracy_fn)
for key, value in model_results.items():
    print(f"\n{key}: {value}")


if (input("save?").lower() == "y"):
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "04_computer_vision.pt"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    torch.save(model.state_dict(), MODEL_SAVE_PATH)


# 5 epochs: cpu time -> 33.463 seconds | gpu time -> 39.872 seconds

# Functionalize training loop
def train_step(model: nn.Module, train_dataloader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device: torch.device = device):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(train_dataloader): # X -> image, y -> label
        X, y = X.to(device), y.to(device)
        y_pred = model(X)

        loss = loss_fn(y_pred)
        train_loss += loss
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")

# Functionalizing testing loop
def test_step(model: nn.Module, test_dataloader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            # test_pred in format of logits
            test_pred = model(X_test.to(device))
            test_loss += loss_fn(test_pred, y_test.to(device))
            test_acc += accuracy_fn(y_test.to(device), test_pred.argmax(dim=1))
            
        # Calculate test loss average per batch
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")