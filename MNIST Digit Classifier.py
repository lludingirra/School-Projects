import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

batch_size = 64
learning_rate = 0.001
num_epochs = 5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class UpdatedCNN(nn.Module):
    def __init__(self):
        super(UpdatedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * 4 * 4, 500),  
            nn.ReLU(),
            nn.Linear(500, 10)  
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = UpdatedCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

model.eval()
correct_count = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct_count += pred.eq(target.view_as(pred)).sum().item()

accuracy = correct_count / len(test_loader.dataset)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
