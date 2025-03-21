import torch
import torchvision
import torchvision.transforms as transforms
from models import MLP_1_ReLU,MLP_1_Sigmoid,MLP_2_ReLU,MLP_2_Sigmoid,CNN_3_ReLU,CNN_3_Sigmoid,CNN_4_ReLU,CNN_4_Sigmoid,CNN_5_ReLU,CNN_5_Sigmoid
from functions import train_and_evaluate, CNN_train_and_evaluate,convert_to_serializable
import json

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device:" , device)

# Hyper-parameters 
num_epochs = 15
batch_size = 50
learning_rate = 0.01

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


# Instantiate the model
model_MLP1_ReLU = MLP_1_ReLU() 
model_MLP2_ReLU = MLP_2_ReLU()
model_CNN3_ReLU = CNN_3_ReLU() 
model_CNN4_ReLU = CNN_4_ReLU()
model_CNN5_ReLU = CNN_5_ReLU()

model_MLP1_Sigmoid = MLP_1_Sigmoid() 
model_MLP2_Sigmoid = MLP_2_Sigmoid()
model_CNN3_Sigmoid = CNN_3_Sigmoid() 
model_CNN4_Sigmoid = CNN_4_Sigmoid()
model_CNN5_Sigmoid = CNN_5_Sigmoid()

####################################################################
#    MLP 1
####################################################################

print("Tranining of MLP_1_ReLU is starting")
results_model_MLP1_ReLU = train_and_evaluate('mpl_1',
                                 model_MLP1_ReLU, 
                                 train_loader, 
                                 test_loader, 
                                 learning_rate, 
                                 num_epochs)
print("MLP_1_ReLU Training and evaluation finished")

print("Tranining of MLP_1_Sigmoid is starting")
results_model_MLP1_Sigmoid = train_and_evaluate('mpl_1',
                                 model_MLP1_Sigmoid, 
                                 train_loader, 
                                 test_loader, 
                                 learning_rate, 
                                 num_epochs)
print("MLP_1_Sigmoid Training and evaluation finished")

results_model_MLP1 = {
    'name': 'MLP1', 
    'relu_loss_curve': results_model_MLP1_ReLU['loss_curve'],
    'sigmoid_loss_curve': results_model_MLP1_Sigmoid['loss_curve'],
    'relu_grad_curve': results_model_MLP1_ReLU['grad_curve'],
    'sigmoid_grad_curve': results_model_MLP1_Sigmoid['grad_curve']
}

result_serializable_MLP1 = convert_to_serializable(results_model_MLP1)
# Save to JSON file
with open('./part4/results/result_MLP1.json', 'w') as f:
    json.dump(result_serializable_MLP1, f, indent=4)



####################################################################
#    MLP 2
####################################################################

print("Tranining of MLP_2_ReLU is starting")
results_model_MLP2_ReLU = train_and_evaluate('mpl_2',
                                 model_MLP2_ReLU, 
                                 train_loader, 
                                 test_loader, 
                                 learning_rate, 
                                 num_epochs)
print("MLP_2_ReLU Training and evaluation finished")

print("Tranining of MLP_2_Sigmoid is starting")
results_model_MLP2_Sigmoid = train_and_evaluate('mpl_2',
                                 model_MLP2_Sigmoid, 
                                 train_loader, 
                                 test_loader, 
                                 learning_rate, 
                                 num_epochs)
print("MLP_2_Sigmoid Training and evaluation finished")

results_model_MLP2 = {
    'name': 'MLP2', 
    'relu_loss_curve': results_model_MLP2_ReLU['loss_curve'],
    'sigmoid_loss_curve': results_model_MLP2_Sigmoid['loss_curve'],
    'relu_grad_curve': results_model_MLP2_ReLU['grad_curve'],
    'sigmoid_grad_curve': results_model_MLP2_Sigmoid['grad_curve']
}

result_serializable_MLP2 = convert_to_serializable(results_model_MLP2)
# Save to JSON file
with open('./part4/results/result_MLP2.json', 'w') as f:
    json.dump(result_serializable_MLP2, f, indent=4)




####################################################################
#    CNN 3
####################################################################

print("Tranining of CNN_3_ReLU is starting")
results_model_CNN3_ReLU = CNN_train_and_evaluate('cnn_3',
                                 model_CNN3_ReLU, 
                                 train_loader, 
                                 test_loader, 
                                 learning_rate, 
                                 num_epochs)
print("CNN_3_ReLU Training and evaluation finished")

print("Tranining of CNN_3_Sigmoid is starting")
results_model_CNN3_Sigmoid = CNN_train_and_evaluate('cnn_3',
                                 model_CNN3_Sigmoid, 
                                 train_loader, 
                                 test_loader, 
                                 learning_rate, 
                                 num_epochs)
print("CNN_3_Sigmoid Training and evaluation finished")

results_model_CNN3 = {
    'name': 'CNN3', 
    'relu_loss_curve': results_model_CNN3_ReLU['loss_curve'],
    'sigmoid_loss_curve': results_model_CNN3_Sigmoid['loss_curve'],
    'relu_grad_curve': results_model_CNN3_ReLU['grad_curve'],
    'sigmoid_grad_curve': results_model_CNN3_Sigmoid['grad_curve']
}

result_serializable_CNN3 = convert_to_serializable(results_model_CNN3)
# Save to JSON file
with open('./part4/results/result_CNN3.json', 'w') as f:
    json.dump(result_serializable_CNN3, f, indent=4)




####################################################################
#    CNN 4
####################################################################

print("Tranining of CNN_4_ReLU is starting")
results_model_CNN4_ReLU = CNN_train_and_evaluate('cnn_4',
                                 model_CNN4_ReLU, 
                                 train_loader, 
                                 test_loader, 
                                 learning_rate, 
                                 num_epochs)
print("CNN_4_ReLU Training and evaluation finished")

print("Tranining of CNN_4_Sigmoid is starting")
results_model_CNN4_Sigmoid = CNN_train_and_evaluate('cnn_4',
                                 model_CNN4_Sigmoid, 
                                 train_loader, 
                                 test_loader, 
                                 learning_rate, 
                                 num_epochs)
print("CNN_4_Sigmoid Training and evaluation finished")

results_model_CNN4 = {
    'name': 'CNN4', 
    'relu_loss_curve': results_model_CNN4_ReLU['loss_curve'],
    'sigmoid_loss_curve': results_model_CNN4_Sigmoid['loss_curve'],
    'relu_grad_curve': results_model_CNN4_ReLU['grad_curve'],
    'sigmoid_grad_curve': results_model_CNN4_Sigmoid['grad_curve']
}

result_serializable_CNN4 = convert_to_serializable(results_model_CNN4)
# Save to JSON file
with open('./part4/results/result_CNN4.json', 'w') as f:
    json.dump(result_serializable_CNN4, f, indent=4)



####################################################################
#    CNN 5
####################################################################

print("Tranining of CNN_5_ReLU is starting")
results_model_CNN5_ReLU = CNN_train_and_evaluate('cnn_5',
                                 model_CNN5_ReLU, 
                                 train_loader, 
                                 test_loader, 
                                 learning_rate, 
                                 num_epochs)
print("CNN_5_ReLU Training and evaluation finished")

print("Tranining of CNN_5_Sigmoid is starting")
results_model_CNN5_Sigmoid = CNN_train_and_evaluate('cnn_5',
                                 model_CNN5_Sigmoid, 
                                 train_loader, 
                                 test_loader, 
                                 learning_rate, 
                                 num_epochs)
print("CNN_5_Sigmoid Training and evaluation finished")

results_model_CNN5 = {
    'name': 'CNN5', 
    'relu_loss_curve': results_model_CNN5_ReLU['loss_curve'],
    'sigmoid_loss_curve': results_model_CNN5_Sigmoid['loss_curve'],
    'relu_grad_curve': results_model_CNN5_ReLU['grad_curve'],
    'sigmoid_grad_curve': results_model_CNN5_Sigmoid['grad_curve']
}

result_serializable_CNN5 = convert_to_serializable(results_model_CNN5)
# Save to JSON file
with open('./part4/results/result_CNN5.json', 'w') as f:
    json.dump(result_serializable_CNN5, f, indent=4)







