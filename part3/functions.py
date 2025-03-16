import torch
import numpy as np

# Training and evaluation function
def train_and_evaluate(model_name, in_model, in_train_loader, in_test_loader, learning_rate ,num_epochs=15, device='cuda'):

  # Device configuration
  model = in_model.to(device)

  # Loss and Optimizer
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

  # For recording loss and accuracy
  training_loss = []
  training_accuracy = []
  validation_accuracy = []

  # Train the model
  n_total_steps = len(in_train_loader)
  for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(in_train_loader):
      # origin shape: [50, 3, 32, 32]
      # resized: [50, 3072]
      images = images.reshape(-1, 3*32*32).to(device)
      labels = labels.to(device)

      # Forward pass and loss calculation
      outputs = model(images)
      loss = criterion(outputs , labels)

      # Backward and optimize
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      running_loss += loss.item()

      # Track training accuracy
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

      # Record loss and accuracy every 10 steps
      if (i + 1) % 10 == 0:
        training_loss.append(running_loss / 10)
        training_accuracy.append(100 * correct / total)
        running_loss = 0.0

    # Calculate validation accuracy
    model.eval()
    with torch.no_grad():
      val_correct = 0
      val_total = 0
      for images, labels in in_test_loader:
        images = images.reshape(-1, 3*32*32).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
      val_acc = 100.0 * val_correct / val_total
      validation_accuracy.append(val_acc)

    print(f'[{epoch + 1}] Training Accuracy: {100 * correct / total:.2f}%, Validation Accuracy: {val_acc:.2f}%')

  print('Finished Training')
  # Test accuracy
  model.eval()
  with torch.no_grad():
    n_correct = 0
    n_samples = len(in_test_loader.dataset)

    for images, labels in in_test_loader:
        images = images.reshape(-1, 3*32*32).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the model: {acc} %')

  # Record the weights of the first layer as numpy array
  first_layer_weights = model.fc1.weight.data.cpu().numpy()


  return {
    'name': model_name,
    'loss_curve': training_loss,
    'train_acc_curve': training_accuracy,
    'val_acc_curve': validation_accuracy,
    'test_acc': acc,
    'weights': first_layer_weights
    }

# Training and evaluation function
def CNN_train_and_evaluate(model_name, in_model, in_train_loader, in_test_loader, learning_rate ,num_epochs=15, device='cuda'):

  # Device configuration
  model = in_model.to(device)

  # Loss and Optimizer
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # For recording loss and accuracy
  training_loss = []
  training_accuracy = []
  validation_accuracy = []

  # Train the model
  n_total_steps = len(in_train_loader)
  for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(in_train_loader):
      images = images.to(device)
      labels = labels.to(device)

      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)

      # Backward and optimize
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      running_loss += loss.item()

      # Track training accuracy
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

      # Record loss and accuracy every 10 steps
      if (i + 1) % 10 == 0:
        training_loss.append(running_loss / 10)
        training_accuracy.append(100 * correct / total)
        running_loss = 0.0
        

      # Calculate validation accuracy
      model.eval()
      with torch.no_grad():
        val_correct = 0
        val_total = 0
        for images, labels in in_test_loader:
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs, 1)
          val_total += labels.size(0)
          val_correct += (predicted == labels).sum().item()
        val_acc = 100.0 * val_correct / val_total
        validation_accuracy.append(val_acc)

    print(f'[{epoch + 1}] Training Accuracy: {100 * correct / total:.2f}%, Validation Accuracy: {val_acc:.2f}%')

  print('Finished Training')
  
  # Test accuracy
  model.eval()
  with torch.no_grad():
    n_correct = 0
    n_samples = len(in_test_loader.dataset)

    for images, labels in in_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the model: {acc} %')

  # Record the weights of the first layer as numpy array
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