### USER
Write a custom implementation of ResNet-18 and its training code using PyTorch.

### ASSISTANT
```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return out + x


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2)
        self.pool = nn.MaxPool2d(3, 2)
        self.block1 = self.make_layer(ResidualBlock, 64, 64)
        self.block2 = self.make_layer(ResidualBlock, 64, 128)
        self.block3 = self.make_layer(ResidualBlock, 128, 256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.avg_pool2d(x, (1, 1))
        x = self.fc(x)
        return x

    def make_layer(self, block, in_channels, out_channels):
        layers = []
        for i in range(2):
            layers.append(block(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)


# Training code

# Load the MNIST dataset from torchvision
import torchvision
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

# Create the DataLoader objects for the training and testing sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create the ResNet18 model
model = ResNet18()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(20):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Flatten MNIST images into a 1D vector
        data = data.view(data.shape[0], -1)

        # Training pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Flatten MNIST images into a 1D vector
            data = data.view(data.shape[0], -1)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy: %f%%' % (100 * correct / total))
```

**Explanation:**

* **Custom ResidualBlock:** Implements the basic building block of ResNet, consisting of two convolutional layers with ReLU activation.
* **Custom ResNet18:** Defines the entire ResNet-18 architecture, including convolutional layers, residual blocks, and fully connected layer for classification.
* **Training Code:** Uses DataLoader to load the MNIST dataset, defines the loss function and optimizer, and iteratively trains the model for 20 epochs.
* **Evaluation:** Measures the accuracy on the test set after each epoch.

**Key Features:**

* Custom implementation of ResNet architecture.
* Gradient descent with stochastic optimization.
* Cross entropy loss function for classification.
* Data augmentation not included (can be added for improved performance).

**Output:**

The output shows the accuracy of the model on the test set after each epoch. The accuracy should gradually increase as the model learns.

**Note:**

* This code requires PyTorch and torchvision libraries for training.
* Adjust the hyperparameters (batch size, learning rate, epochs) as needed.
* Consider adding data augmentation techniques to enhance model performance.