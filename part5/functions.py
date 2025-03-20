import torch
import numpy as np


def CNN_train_and_evaluate(model_name, in_model, in_train_loader, in_test_loader, learning_rate, num_epochs=15, target_val_accuracy=None, reinitialize=True ,device='cuda'):
  model = in_model.to(device)


  if reinitialize:
    # He initialization for ReLU activations for applicable layers
    def init_weights(m):
      if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
          m.bias.data.fill_(0.0)
    model.apply(init_weights)

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)

  loss_curve = []
  validation_accuracy = []

  # Preload all test data once
  test_images, test_labels = [], []
  for images, labels in in_test_loader:
    test_images.append(images.to(device))
    test_labels.append(labels.to(device))
  test_images = torch.cat(test_images)
  test_labels = torch.cat(test_labels)
  test_batch_size = in_test_loader.batch_size  # Preserve original batch size

  early_stop = False  # Flag to check early stopping condition

  for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()

    for i, (images, labels) in enumerate(in_train_loader):
      images, labels = images.to(device), labels.to(device)
      images = images + 0.007 * torch.randn_like(images)  # Adversarial Training: adding noise to images

      # Forward + backward
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      running_loss += loss.item()

      # Validation every 10 batches
      if (i + 1) % 10 == 0:
        avg_loss = running_loss / 10
        loss_curve.append(avg_loss)
        running_loss = 0.0

        model.eval()
        val_correct = 0
        with torch.no_grad():
          for batch_start in range(0, len(test_images), test_batch_size):
            batch_images = test_images[batch_start:batch_start+test_batch_size]
            batch_labels = test_labels[batch_start:batch_start+test_batch_size]
            outputs = model(batch_images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == batch_labels).sum().item()
        val_acc = 100.0 * val_correct / len(test_labels)
        validation_accuracy.append(val_acc)
                
        # Check if early stopping criterion is met
        if target_val_accuracy is not None and val_acc >= target_val_accuracy:
          print(f"Early stopping triggered: Validation accuracy {val_acc:.2f}% reached target of {target_val_accuracy}%")
          early_stop = True
          break  # Exit inner loop

        model.train()

    print(f'[{epoch+1}/{num_epochs}] Epoch completed. Last Validation Accuracy: {val_acc:.2f}%')
        
    if early_stop:
      break  # Exit outer loop if early stopping was triggered

  # Return only the loss curve and validation accuracy data
  return {
    'loss_curve': loss_curve,
    'validation_accuracy': validation_accuracy
  }


def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert tensors to lists
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}  # Recursively convert dicts
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]  # Recursively convert lists
    return obj  # Return the object as is if it's already serializable