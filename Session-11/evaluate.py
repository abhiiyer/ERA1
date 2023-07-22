import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from utils import test, get_cifar10_data

# Function to get misclassified images
def get_misclassified_images(model, test_loader, device, num_images=10):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    correct_labels = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            misclassified_mask = (predicted != labels)

            misclassified_images.append(images[misclassified_mask])
            misclassified_labels.append(predicted[misclassified_mask])
            correct_labels.append(labels[misclassified_mask])

    misclassified_images = torch.cat(misclassified_images)[:num_images]
    misclassified_labels = torch.cat(misclassified_labels)[:num_images]
    correct_labels = torch.cat(correct_labels)[:num_images]

    return misclassified_images, misclassified_labels, correct_labels

# Function to get GradCAM outputs on misclassified images
def get_gradcam_outputs(model, images, target_layers, device):
    model.eval()
    activations = []
    grads = {}

    def hook_fn(module, input, output):
        activations.append(output)
    def backward_hook_fn(module, grad_input, grad_output):
        grads['value'] = grad_output[0].detach()

    hooks = []
    for layer in target_layers:
        hooks.append(layer.register_forward_hook(hook_fn))
        hooks.append(layer.register_backward_hook(backward_hook_fn))

    outputs = model(images.to(device))
    one_hot_output = torch.zeros_like(outputs, dtype=torch.float)
    one_hot_output[:, outputs.argmax(dim=1)] = 1.0

    model.zero_grad()
    outputs.backward(gradient=one_hot_output.to(device), retain_graph=True)

    for hook in hooks:
        hook.remove()

    if 'value' not in grads:
        raise ValueError("Gradients not found. Make sure the model was trained with gradients enabled.")
    
    pooled_grads = grads['value'].mean(dim=[2, 3]).cpu().numpy()
    activations = activations[-1].squeeze().detach().cpu().numpy()

    heatmap = np.mean(pooled_grads[..., None, None] * activations, axis=1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap
    
    
# Function to plot misclassified images with GradCAM overlay
def plot_gradcam_images(model, test_loader, device, classes, target_layers, num_images=10):
    model.eval()
    misclassified_images, predicted_labels, correct_labels = get_misclassified_images(model, test_loader, device, num_images=num_images)

    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(8, 18))

    for idx, (image, pred_label, correct_label) in enumerate(zip(misclassified_images, predicted_labels, correct_labels)):
        heatmap = get_gradcam_outputs(model, image.unsqueeze(0), target_layers, device)
        heatmap = cv2.resize(heatmap, (image.size(-1), image.size(-2)))
        heatmap = np.uint8(255 * heatmap)

        image_np = np.transpose(image.cpu().numpy(), (1, 2, 0))
        image_np = (image_np * 0.5) + 0.5
        image_np = np.uint8(255 * image_np)

        ax1, ax2 = axes[idx]
        ax1.imshow(image_np)
        ax1.axis('off')
        ax1.set_title(f'Pred: {classes[pred_label.item()]}')
        ax2.imshow(image_np)
        ax2.imshow(heatmap, alpha=0.6, cmap='jet')
        ax2.axis('off')
        ax2.set_title(f'True: {classes[correct_label.item()]} (GradCAM)')

    plt.tight_layout()
    plt.show()
