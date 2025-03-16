import torch
import numpy as np

def train_and_evaluate(model_name, in_model, in_train_loader, in_test_loader, learning_rate, num_epochs=15, device='cuda'):
  model = in_model.to(device)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  training_loss = []
  training_accuracy = []
  validation_accuracy = []

  # Preload and flatten test data
  test_images, test_labels = [], []
  for images, labels in in_test_loader:
    test_images.append(images.reshape(-1, 3*32*32).to(device))
    test_labels.append(labels.to(device))
  test_images = torch.cat(test_images, dim=0)
  test_labels = torch.cat(test_labels, dim=0)
  test_batch_size = in_test_loader.batch_size  # Use the same batch size as the test loader

  n_total_steps = len(in_train_loader)
  for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()

    for i, (images, labels) in enumerate(in_train_loader):
      images = images.reshape(-1, 3*32*32).to(device)
      labels = labels.to(device)

      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)

      # Backward pass and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

      # Log training metrics every 10 steps
      if (i + 1) % 10 == 0:
        training_loss.append(running_loss / 10)
        training_accuracy.append(100 * correct / total)
        running_loss = 0.0

        # Calculate validation accuracy using preloaded data
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
          for i_batch in range(0, len(test_images), test_batch_size):
            batch_images = test_images[i_batch:i_batch+test_batch_size]
            batch_labels = test_labels[i_batch:i_batch+test_batch_size]
            outputs = model(batch_images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_labels.size(0)
            val_correct += (predicted == batch_labels).sum().item()
          val_acc = 100.0 * val_correct / val_total
        validation_accuracy.append(val_acc)
        model.train()

    # Print epoch statistics
    epoch_train_acc = 100 * correct / total
    print(f'[{epoch + 1}] Training Accuracy: {epoch_train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%')

  # Final evaluation using preloaded test data
  model.eval()
  n_correct = 0
  with torch.no_grad():
    for i_batch in range(0, len(test_images), test_batch_size):
      batch_images = test_images[i_batch:i_batch+test_batch_size]
      batch_labels = test_labels[i_batch:i_batch+test_batch_size]
      outputs = model(batch_images)
      _, predicted = torch.max(outputs.data, 1)
      n_correct += (predicted == batch_labels).sum().item()
  acc = 100.0 * n_correct / len(test_labels)
  print(f'Accuracy of the model: {acc:.2f} %')

  first_layer_weights = model.fc1.weight.data.cpu().numpy()

  return {
    'name': model_name,
    'loss_curve': training_loss,
    'train_acc_curve': training_accuracy,
    'val_acc_curve': validation_accuracy,
    'test_acc': acc,
    'weights': first_layer_weights
  }

def CNN_train_and_evaluate(model_name, in_model, in_train_loader, in_test_loader, learning_rate, num_epochs=15, device='cuda'):
  model = in_model.to(device)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  training_loss = []
  training_accuracy = []
  validation_accuracy = []

  # Preload all test data once
  test_images, test_labels = [], []
  for images, labels in in_test_loader:
    test_images.append(images.to(device))
    test_labels.append(labels.to(device))
  test_images = torch.cat(test_images)
  test_labels = torch.cat(test_labels)
  test_batch_size = in_test_loader.batch_size  # Preserve original batch size

  for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()

    for i, (images, labels) in enumerate(in_train_loader):
      images, labels = images.to(device), labels.to(device)

      # Forward + backward
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      # Update metrics
      running_loss += loss.item()
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

      # Validation every 10 batches
      if (i + 1) % 10 == 0:
        # Store training metrics
        training_loss.append(running_loss / 10)
        training_accuracy.append(100 * correct / total)
        running_loss = 0.0

        # Fast validation using preloaded data
        model.eval()
        val_correct = 0
        with torch.no_grad():
          # Process in original batch sizes
          for batch_start in range(0, len(test_images), test_batch_size):
            batch_images = test_images[batch_start:batch_start+test_batch_size]
            batch_labels = test_labels[batch_start:batch_start+test_batch_size]
            outputs = model(batch_images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == batch_labels).sum().item()
                        
        val_acc = 100.0 * val_correct / len(test_labels)
        validation_accuracy.append(val_acc)
        model.train()

    # Epoch statistics
    epoch_acc = 100 * correct / total
    print(f'[{epoch+1}] Train Acc: {epoch_acc:.2f}%, Val Acc: {val_acc:.2f}%')

  # Final evaluation with preloaded data
  model.eval()
  final_correct = 0
  with torch.no_grad():
    for batch_start in range(0, len(test_images), test_batch_size):
      batch_images = test_images[batch_start:batch_start+test_batch_size]
      batch_labels = test_labels[batch_start:batch_start+test_batch_size]
      outputs = model(batch_images)
      final_correct += (torch.max(outputs, 1)[1] == batch_labels).sum().item()
    
  acc = 100.0 * final_correct / len(test_labels)
  print(f'Final Accuracy: {acc:.2f}%')

  first_layer_weights = model.conv1.weight.data.cpu().numpy()

  return {
    'name': model_name,
    'loss_curve': training_loss,
    'train_acc_curve': training_accuracy,
    'val_acc_curve': validation_accuracy,
    'test_acc': acc,
    'weights': first_layer_weights
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