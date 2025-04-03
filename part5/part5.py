import torch
import torchvision
import torchvision.transforms as transforms
from models import CNN_5, CNN_4
from functions import CNN_train_and_evaluate,convert_to_serializable
import json

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device:" , device)

# Hyper-parameters 
num_epochs = 20
batch_size = 50



#################################################################################
# Transformations
#################################################################################

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

# Load the dataset (assuming it's a custom dataset with 100,000 images)
full_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True, 
                                            download=True, 
                                            transform=transform)

# Compute split sizes
train_size = int(0.9 * len(full_dataset))   # 90% training
test_size = len(full_dataset) - train_size  # 10% testing

# Split dataset
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)




#################################################################################
# Instantiate the model
#################################################################################

model_CNN5 = CNN_5()
model_CNN5_scheduled = CNN_5()




#################################################################################
# Testing Dİfferent Learning Rates
#################################################################################
print("Tranining of CNN5 is starting with lr = 0.1")
result_CNN5_1 = CNN_train_and_evaluate('cnn_5',
                                     model_CNN5, 
                                     train_loader, 
                                     test_loader, 
                                     0.1,         # Learning Rate
                                     20,          # Number of epochs
                                     100,         # Target Validation
                                     True,        # Reinitialize Weights
                                     device)
print("CNN_5 Training and evaluation finished with lr = 0.1")

print("Tranining of CNN5 is starting with lr = 0.01")
result_CNN5_01 = CNN_train_and_evaluate('cnn_5',
                                     model_CNN5, 
                                     train_loader, 
                                     test_loader, 
                                     0.01,        # Learning Rate   
                                     20,          # Number of epochs
                                     100,         # Target Validation
                                     True,        # Reinitialize Weights
                                     device)
print("CNN_5 Training and evaluation finishedwith lr = 0.01")

print("Tranining of CNN5 is starting with lr = 0.001")
result_CNN5_001 = CNN_train_and_evaluate('cnn_5',
                                     model_CNN5, 
                                     train_loader, 
                                     test_loader, 
                                     0.001,       # Learning Rate
                                     20,          # Number of epochs
                                     100,         # Target Validation
                                     True,        # Reinitialize Weights
                                     device)
print("CNN_5 Training and evaluation finished with lr = 0.001")

results_model_CNN5 = {
    'name': 'CNN5', 
    'loss_curve_1': result_CNN5_1['loss_curve'],
    'loss_curve_01': result_CNN5_01['loss_curve'],
    'loss_curve_001': result_CNN5_001['loss_curve'],
    'val_acc_curve_1': result_CNN5_1['validation_accuracy'],
    'val_acc_curve_01': result_CNN5_01['validation_accuracy'],
    'val_acc_curve_001': result_CNN5_001['validation_accuracy']
}

result_serializable_CNN5 = convert_to_serializable(results_model_CNN5)
# Save to JSON file
with open('./part5/results/result_CNN5_diff_lr.json', 'w') as f:
    json.dump(result_serializable_CNN5, f, indent=4)






#################################################################################
# Scheduling Learning First Try
#################################################################################

print("Tranining of Scheduling CNN5 is starting with lr = 0.1")
result_CNN5_sch_part1 = CNN_train_and_evaluate('cnn_5',
                                     model_CNN5_scheduled, 
                                     train_loader, 
                                     test_loader, 
                                     0.1,         # Learning Rate
                                     30,          # Number of Epochs
                                     65,          # Target Validancy
                                     True,        # Reinitialize Weights
                                     device)
print("CNN_5 Training and evaluation finished with lr = 0.1, without achieved desired validancy")

print("Tranining of CNN5 is starting with lr = 0.01")
result_CNN5_sch_part2 = CNN_train_and_evaluate('cnn_5',
                                     model_CNN5_scheduled, 
                                     train_loader, 
                                     test_loader, 
                                     0.01,        # Learning Rate
                                     30,          # Number of Epochs
                                     70,         # Target Validancy
                                     False,       # Reinitialize Weights
                                     device)
print("CNN_5 Training and evaluation finished with lr = 0.01, without achieved desired validancy")

print("Tranining of CNN5 is starting with lr = 0.001")
result_CNN5_sch_part2 = CNN_train_and_evaluate('cnn_5',
                                     model_CNN5_scheduled, 
                                     train_loader, 
                                     test_loader, 
                                     0.001,        # Learning Rate
                                     30,          # Number of Epochs
                                     100,         # Target Validancy
                                     False,       # Reinitialize Weightss
                                     device)
print("CNN_5 Training and evaluation finishedwith lr = 0.001")


results_model_CNN5_sch = []
for result in [result_CNN5_sch_part1, result_CNN5_sch_part2]:
  results_model_CNN5_sch += result['validation_accuracy']

result_serializable_CNN5_sch = convert_to_serializable(results_model_CNN5_sch)
# Save to JSON file
with open('./part5/results/result_CNN5_sch_1.json', 'w') as f:
    json.dump(result_serializable_CNN5_sch, f, indent=4)
