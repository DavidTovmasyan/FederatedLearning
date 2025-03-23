import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from models import ClientServer, CentralServer


def federated_learning(num_clients=10, num_rounds=100, local_epochs=1, batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=transform)

    test_indices = np.arange(len(testset))
    np.random.shuffle(test_indices)
    split_idx = len(test_indices) // 2

    fine_tune_set = Subset(testset, test_indices[:split_idx])
    eval_set = Subset(testset, test_indices[split_idx:])

    fine_tune_loader = DataLoader(fine_tune_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

    indices = np.arange(len(trainset))
    np.random.shuffle(indices)
    client_splits = np.array_split(indices, num_clients)

    client_dataloaders = {
        i: DataLoader(Subset(trainset, client_splits[i]), batch_size=batch_size, shuffle=True)
        for i in range(num_clients)
    }

    server = CentralServer(num_classes=10, device=device)
    global_model = server.global_model

    for round in range(num_rounds):
        print(f"=== Round {round + 1} ===")

        client_models = []
        for client_id in range(num_clients):
            client = ClientServer(global_model)
            client.train(client_dataloaders[client_id], epochs=local_epochs, device=device)
            client_models.append(client.get_model())

        server.aggregate(client_models)

        global_model = server.global_model
        server.fine_tune(fine_tune_loader, epochs=1, device=device)

        test_accuracy = evaluate(global_model, eval_loader, device=device)
        print(f"Round {round + 1} - Global Model Test Accuracy: {test_accuracy:.2f}%\n")

        if test_accuracy > 81:
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
    global_model = federated_learning()
    torch.save(global_model.state_dict(), "./final_models/FedAvg/global_model.pth")

if __name__ == "__main__":
    main()