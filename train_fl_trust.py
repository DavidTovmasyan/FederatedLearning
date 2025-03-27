import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split

from models_fl_trust import CentralServerFLTrust, ClientServerFLTrust


def federated_learning_fltrust(num_clients=10, num_rounds=100, local_epochs=1, batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=transform)

    test_indices = np.arange(len(testset))
    np.random.shuffle(test_indices)
    split_idx = len(test_indices) // 2

    trusted_set = Subset(testset, test_indices[:split_idx])
    test_set = Subset(testset, test_indices[split_idx:])

    trusted_loader = DataLoader(trusted_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)



    indices = np.arange(len(trainset))
    np.random.shuffle(indices)
    client_splits = np.array_split(indices, num_clients)

    client_dataloaders = {}
    validation_dataloaders = {}

    for i in range(num_clients):
        train_size = int(0.8 * len(client_splits[i]))
        val_size = len(client_splits[i]) - train_size
        train_subset, val_subset = random_split(client_splits[i], [train_size, val_size])
        client_dataloaders[i] = DataLoader(Subset(trainset, train_subset), batch_size=batch_size, shuffle=True)
        validation_dataloaders[i] = DataLoader(Subset(trainset, val_subset), batch_size=batch_size, shuffle=False)

    server = CentralServerFLTrust(num_classes=10, device=device, root_data=trusted_loader)
    global_model = server.global_model


    for round in range(num_rounds):
        print(f"=== Round {round + 1} ===")
        attackers = random.sample(range(num_clients), 4)  # 4 out of 10 clients are attackers
        print(f"Attackers: {attackers}")
        client_models = []
        total_val_loss = 0.0
        total_val_acc = 0.0

        for client_id in range(num_clients):
            is_attacker = client_id in attackers
            client = ClientServerFLTrust(global_model, is_attacker=is_attacker, device="cuda")
            print(f"Training the client: {client_id}")
            client.train(client_dataloaders[client_id], epochs=local_epochs, device=device)
            print(f"Validating the client: {client_id}")
            val_loss, val_acc = client.validate(validation_dataloaders[client_id], device=device)
            total_val_loss += val_loss
            total_val_acc += val_acc
            client_models.append(client.get_model())
        print("Aggreagation...")
        server.aggregate(client_models=client_models, ground_truth_loader=trusted_loader,
                         local_epochs=local_epochs, device="cuda")
        avg_val_loss = total_val_loss / num_clients
        avg_val_acc = total_val_acc / num_clients
        print(f"Round {round + 1} - Average Validation Loss: {avg_val_loss:.4f}, Average Validation Accuracy: {avg_val_acc:.2f}%\n")

        print("Evaluating...")
        test_acc = evaluate(global_model, test_loader, device=device)
        print(f"Round {round + 1} - Global Model Test Accuracy: {test_acc:.2f}%\n")

        if test_acc >= 81:
            print("Stopping early as test accuracy reached threshold.")
            break

    return global_model

def evaluate(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def main():
    fl_trust_model = federated_learning_fltrust()
    torch.save(fl_trust_model.state_dict(), "./final_models/FLTrust/FLTrust.pth")

if __name__ == '__main__':
    main()
