import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, train_loader, val_loader, epochs, optimizer, loss_fn, device, patience=5):
    best_loss = float('inf')
    epochs_no_improve = 0
    model.to(device)

    for epoch in range(epochs):
        model.train()  
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            labels = labels.float().unsqueeze(1)
            
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            predicted = torch.sigmoid(outputs) >= 0.5
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}, Accuracy: {accuracy}')
        
        val_loss = validate_model(model, val_loader, loss_fn, device)
        print(f'Validation Loss: {val_loss}')
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model, '/data/models/model.pth')
            print(f"Model is saved by early stopping; epoch: {epoch+1}, loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping!')
                break
        

def validate_model(model, val_loader, loss_fn, device):
    model.eval()  
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1) 

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(val_loader)
    return average_loss

