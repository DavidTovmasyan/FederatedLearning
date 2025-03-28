import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class CentralServer:
    def __init__(self, num_classes=10, device="cpu"):
        self.global_model = Model(num_classes)
        self.global_model.to(device)

    def aggregate(self, client_models):
        with torch.no_grad():
            global_params = {name: torch.zeros_like(param) for name, param in self.global_model.state_dict().items()}

            for client_model in client_models:
                for name, param in client_model.state_dict().items():
                    global_params[name] += param

            for name in global_params:
              global_params[name] = (global_params[name].float() / len(client_models)).to(self.global_model.state_dict()[name].dtype)


            self.global_model.load_state_dict(global_params)

class ClientServer:
    def __init__(self, global_model, device="cpu"):
        self.model = Model()
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(device)

    def train(self, train_loader, epochs=1, lr=0.001, device="cpu"):
        self.model.to(device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

    def validate(self, val_loader, device="cpu"):
        self.model.to(device)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, total_samples, correct = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total_samples
        return total_loss / total_samples, accuracy

    def get_model(self):
        return self.model