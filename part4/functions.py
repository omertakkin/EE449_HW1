import torch
import numpy as np

def train_and_evaluate(model_name, in_model, in_train_loader, in_test_loader, learning_rate, num_epochs=15, device='cuda'):
  model = in_model.to(device)

  # He initialization, good for ReLU
  def init_weights(m):
    if isinstance(m, torch.nn.Linear):
      torch.nn.init.kaiming_uniform_(m.weight)
      if m.bias is not None:
        m.bias.data.fill_(0.0)

  model.apply(init_weights)

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)

  training_loss = []
  grad_curve = []

  n_total_steps = len(in_train_loader)
  for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()

    for i, (images, labels) in enumerate(in_train_loader):
      images = images.reshape(-1, 3*32*32).to(device)
      images = images + 0.007 * torch.randn_like(images)  # Adversarial Training
      labels = labels.to(device)

      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)

      # Backward pass
      optimizer.zero_grad()
      loss.backward()

      # Compute gradient norm
      total_grad_norm = 0.0
      for param in model.parameters():
        if param.grad is not None:
          total_grad_norm += param.grad.norm().item() ** 2
      total_grad_norm = total_grad_norm ** 0.5  # Square root of sum of squares

      optimizer.step()

      running_loss += loss.item()

      # Log loss and gradient every 10 steps
      if (i + 1) % 10 == 0:
        training_loss.append(running_loss / 10)
        grad_curve.append(total_grad_norm)
        running_loss = 0.0

    print(f"Epoch [{epoch+1}/{num_epochs}] completed.")

  return {
    'loss_curve': training_loss,
    'grad_curve': grad_curve
  }


def CNN_train_and_evaluate(model_name, in_model, in_train_loader, in_test_loader, learning_rate, num_epochs=15, device='cuda'):
  model = in_model.to(device)

  # He initialization for applicable layers
  def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
      torch.nn.init.kaiming_uniform_(m.weight)
      if m.bias is not None:
        m.bias.data.fill_(0.0)
  model.apply(init_weights)

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)

  training_loss = []
  grad_curve = []

  for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()

    for i, (images, labels) in enumerate(in_train_loader):
      images, labels = images.to(device), labels.to(device)
      images = images + 0.007 * torch.randn_like(images)  # Adversarial Training

      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)

      # Backward pass and gradient computation
      optimizer.zero_grad()
      loss.backward()

      # Compute gradient norm over all parameters
      total_grad_norm = 0.0
      for param in model.parameters():
        if param.grad is not None:
          total_grad_norm += param.grad.norm().item() ** 2
      total_grad_norm = total_grad_norm ** 0.5

      optimizer.step()

      running_loss += loss.item()

      # Record metrics every 10 batches
      if (i + 1) % 10 == 0:
        avg_loss = running_loss / 10
        training_loss.append(avg_loss)
        grad_curve.append(total_grad_norm)
        running_loss = 0.0

    print(f"Epoch [{epoch+1}/{num_epochs}] completed.")

  # Return only the loss and gradient curves
  return {
    'loss_curve': training_loss,
    'grad_curve': grad_curve
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