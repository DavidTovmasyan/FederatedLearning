import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from models import Model


class CentralServerFLTrust:
    def __init__(self, root_data, num_classes=10, device="cpu"):
        self.global_model = Model(num_classes)
        self.global_model.to(device)
        self.trusted_model = Model(num_classes)  # Trusted model for FLTrust
        self.trusted_model.to(device)
        self.root_data = root_data

    def aggregate(self, client_models, ground_truth_loader, local_epochs, lr=0.001, alpha=1.0, device="cpu"):
        with torch.no_grad():
            global_params = {name: param.clone().detach() for name, param in self.global_model.state_dict().items()}

        # Step 1: Train the Trusted Model on Ground Truth Data
        self.trusted_model.load_state_dict(self.global_model.state_dict())
        self.trusted_model.train()
        optimizer = optim.AdamW(self.trusted_model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(local_epochs):
            for images, labels in ground_truth_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.trusted_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        self.trusted_model.eval()
        trusted_params = {name: param.clone().detach() for name, param in self.trusted_model.state_dict().items()}

        # Compute model updates: g0 = w_t - w
        trusted_updates = {name: trusted_params[name] - global_params[name] for name in trusted_params}

        # Step 2: Compute Trust Scores & Normalize Client Updates
        ts_values = []
        client_updates_normalized = []

        for client_model in client_models:
            client_params = client_model.state_dict()

            # Compute model updates: g_i = w_r - w
            client_updates = {name: client_params[name] - global_params[name] for name in client_params}

            # Compute Trust Score (cosine similarity between g_i and g0)
            ts_i = sum(F.cosine_similarity(client_updates[name].float().flatten(),
                                          trusted_updates[name].float().flatten(), dim=0)
                      for name in client_updates)

            ts_i = F.relu(ts_i)  # Apply ReLU to similarity score
            ts_values.append(ts_i)

            # Compute norm of the entire trusted update
            trusted_update_norm = torch.norm(torch.cat([v.view(-1).float() for v in trusted_updates.values()]))
            client_update_norm = torch.norm(torch.cat([v.view(-1).float() for v in client_updates.values()]))

            # Scale client update
            scaling_factor = trusted_update_norm / (client_update_norm + 1e-6)  # Avoid division by zero
            normalized_update = {name: scaling_factor * client_updates[name] for name in client_updates}

            client_updates_normalized.append(normalized_update)

        # Step 3: Aggregate Client Updates
        total_ts = sum(ts_values)
        aggregated_update = {name: torch.zeros_like(global_params[name]) for name in global_params}

        for i, client_update in enumerate(client_updates_normalized):
            weight = ts_values[i] / total_ts if total_ts != 0 else 0
            for name, update in client_update.items():
                aggregated_update[name] += (weight * update.float()).to(aggregated_update[name].dtype)

        # Step 4: Apply Update to Global Model: w ← w + α * g
        for name in global_params:
            global_params[name] += (alpha * aggregated_update[name]).to(global_params[name].dtype)

        self.global_model.load_state_dict(global_params)


class ClientServerFLTrust:
    def __init__(self, global_model, is_attacker=False, device="cpu"):
        self.model = Model()
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(device)
        self.is_attacker = is_attacker

    def train(self, dataloader, epochs=5, lr=0.001, device="cpu"):
        self.model.to(device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)

                if self.is_attacker:
                    labels = torch.randint(0, 10, labels.shape, device=device)  # Randomly flip labels

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

