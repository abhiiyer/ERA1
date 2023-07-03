import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

def train(model, device, train_loader, optimizer, criterion, scheduler):
    model.train()
    train_loss = 0
    total_train_samples = 0
    correct_train_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Compute train accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train_samples += labels.size(0)
        correct_train_samples += (predicted == labels).sum().item()

        train_loss += loss.item()

    train_accuracy = correct_train_samples / total_train_samples
    return train_loss / len(train_loader), train_accuracy

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    total_test_samples = 0
    correct_test_samples = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test_samples += labels.size(0)
            correct_test_samples += (predicted == labels).sum().item()

            # Compute per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    test_accuracy = correct_test_samples / total_test_samples
    return test_loss / len(test_loader), test_accuracy, class_correct, class_total

def get_max_test_accuracy(test_acc):
    max_acc = np.max(test_acc)
    max_epoch = np.argmax(test_acc)
    return max_acc, max_epoch
