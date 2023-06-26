# import torch
# import matplotlib.pyplot as plt


# def train(model, device, train_loader, optimizer, criterion, epochs):
#     train_losses = []
#     train_accuracies = []

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0

#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#         epoch_loss = running_loss / len(train_loader)
#         epoch_accuracy = correct / total * 100

#         train_losses.append(epoch_loss)
#         train_accuracies.append(epoch_accuracy)

#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.2f}%")

#     return train_losses, train_accuracies


# def test(model, device, test_loader, criterion):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     class_correct = [0] * 10
#     class_total = [0] * 10

#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             loss = criterion(output, target)
#             test_loss += loss.item()
#             _, predicted = output.max(1)
#             total += target.size(0)
#             correct += predicted.eq(target).sum().item()

#             # Calculate per-class accuracy
#             for i in range(10):
#                 class_total[i] += target[target == i].size(0)
#                 class_correct[i] += predicted[target == i].eq(target[target == i]).sum().item()

#     test_loss /= len(test_loader.dataset)
#     accuracy = 100.0 * correct / total
#     class_accuracy = [100.0 * class_correct[i] / class_total[i] for i in range(10)]

#     print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
#     #print("Per-Class Accuracy:")
#     #for i in range(10):
#     #    print(f"Class {i}: {class_accuracy[i]:.2f}%")

#     return test_loss, accuracy, class_accuracy


# import matplotlib.pyplot as plt

# def plot_learning_curve(train_losses, test_loss, train_accuracies, test_accuracy, class_accuracy, class_names):
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses, label="Train")
#     plt.plot(test_loss, label="Test")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(train_accuracies, label="Train")
#     plt.plot(test_accuracy, label="Test")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.legend()

#     plt.tight_layout()

#    # Plot per-class accuracy
#     plt.figure(figsize=(8, 6))  # Increase the figure size to accommodate class names
#     plt.bar(range(10), class_accuracy, tick_label=class_names)  # Use class_names as tick labels
#     plt.xlabel("Class")
#     plt.ylabel("Accuracy")
#     plt.title("Per-Class Accuracy")
    
#     # Add text labels for accuracy on top of each bar
#     for i in range(10):
#         plt.text(i, class_accuracy[i], f"{class_accuracy[i]:.2f}%", ha="center", va="bottom")

#     plt.show()


import torch
import matplotlib.pyplot as plt


def train(model, device, train_loader, optimizer, criterion, epochs, test_loader):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        #print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.3f}, Train Accuracy: {epoch_accuracy:.2f}%")

        # Evaluate on test set
        test_loss, test_accuracy, _ = test(model, device, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        #print(f"Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.3f}, Train Accuracy: {epoch_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


    return train_losses, test_losses, train_accuracies, test_accuracies


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Calculate per-class accuracy
            for i in range(10):
                class_total[i] += target[target == i].size(0)
                class_correct[i] += predicted[target == i].eq(target[target == i]).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / total
    class_accuracy = [100.0 * class_correct[i] / class_total[i] for i in range(10)]

    # Print test accuracy
    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    return test_loss, accuracy, class_accuracy


import matplotlib.pyplot as plt

def plot_learning_curve(train_losses, test_losses, train_accuracies, test_accuracies, class_accuracy, class_names):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train")
    plt.plot(test_accuracies, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

   # Plot per-class accuracy
    plt.figure(figsize=(8, 6))  # Increase the figure size to accommodate class names
    plt.bar(range(10), class_accuracy, tick_label=class_names)  # Use class_names as tick labels
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    
    # Add text labels for accuracy on top of each bar
    for i in range(10):
        plt.text(i, class_accuracy[i], f"{class_accuracy[i]:.2f}%", ha="center", va="bottom")

    plt.show()




