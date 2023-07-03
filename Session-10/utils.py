import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt


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

def plot_loss_accuracy(train_losses, train_acc, test_losses, test_acc):
    """Plot the training and test loss/accuracy curves."""
    epochs = len(train_losses)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_acc, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def show_misclassified_images(device, model, test_loader, class_labels, num_samples=10):
    """Display the misclassified images with their predicted and true labels."""
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            incorrect_indices = (predicted != labels).nonzero()[:, 0]
            for idx in incorrect_indices:
                misclassified_images.append(images[idx].cpu())
                misclassified_labels.append(labels[idx].cpu())
                misclassified_preds.append(predicted[idx].cpu())

            if len(misclassified_images) >= num_samples:
                break

    num_cols = 5
    num_rows = (num_samples + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    fig.suptitle('Misclassified Images')
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            image = misclassified_images[i]
            label = misclassified_labels[i].item()
            pred = misclassified_preds[i].item()
            ax.imshow(image.permute(1, 2, 0).numpy())
            ax.axis('off')
            ax.set_title(f'True: {class_labels[label]}\nPred: {class_labels[pred]}')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()