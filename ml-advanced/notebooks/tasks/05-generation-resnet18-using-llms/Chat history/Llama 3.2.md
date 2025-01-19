### USER
Write a custom implementation of ResNet-18 and its training code using PyTorch.

### ASSISTANT
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256*4*4, 512)
        self.relu2 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        
        out = self.conv4(out)
        out = self.bn4(out)
        
        out = out.view(-1, 256*4*4)
        out = self.fc1(out)
        out = self.relu2(out)
        
        out = self.fc2(out)
        return out

class ResNet18Trainer:
    def __init__(self, model, device, train_loader, val_loader, num_epochs=10):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        
        # Define the loss function and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}")
            running_loss = 0.0
            
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                
                # Backward and optimize
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Training loss: {running_loss / (i + 1)}")
        
        return self

    def validate(self):
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                
                total_correct += (predicted == labels).sum().item()
                total_samples += len(labels)
        
        accuracy = 100.0 * total_correct / total_samples
        print(f"Validation accuracy: {accuracy:.2f}%")

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Set up the data loaders and transforms
    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='./data/validation', transform=transform)
    
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create the model and trainer instances
    model = ResNet18().to(device)
    trainer = ResNet18Trainer(model, device, train_loader, val_loader)
    
    # Train the model
    trainer.train()
    trainer.validate()