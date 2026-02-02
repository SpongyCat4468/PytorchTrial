import torch.nn as nn
import torch
import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


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

fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
class_items = {value: key for key, value in train_data.class_to_idx.items()}
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_items[label])
    plt.axis(False)
plt.show()