import torch
import torchvision
import torchvision.transforms as transforms
from models import MLP_1,MLP_2,CNN_3,CNN_4,CNN_5
from functions import train_and_evaluate, CNN_train_and_evaluate,convert_to_serializable
import json

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device:" , device)

# Hyper-parameters 
num_epochs = 15
batch_size = 50
learning_rate = 0.001

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
model_MLP1 = MLP_1() 
model_MLP2 = MLP_2()
model_CNN3 = CNN_3() 
model_CNN4 = CNN_4()
model_CNN5 = CNN_5()

print("Tranining of MLP1 is starting")
result_MLP1 = train_and_evaluate('mpl_1',
                                 model_MLP1, 
                                 train_loader, 
                                 test_loader, 
                                 learning_rate, 
                                 num_epochs)
print("MLP_1 Training and evaluation finished")
PATH = './part3/results/MLP_1.pth'
torch.save(model_MLP1.state_dict(), PATH)

result_serializable_MLP1 = convert_to_serializable(result_MLP1)
# Save to JSON file
with open('./part3/results/result_MLP1.json', 'w') as f:
    json.dump(result_serializable_MLP1, f, indent=4)


print("Tranining of MLP2 is starting")
result_MLP2 = train_and_evaluate('mlp_2',
                                 model_MLP2, 
                                 train_loader, 
                                 test_loader, 
                                 learning_rate, 
                                 num_epochs)
print("MLP_2 Training and evaluation finished")
PATH = './part3/results/MLP_2.pth'
torch.save(model_MLP2.state_dict(), PATH)

result_serializable_MLP2 = convert_to_serializable(result_MLP2)
# Save to JSON file
with open('./part3/results/result_MLP2.json', 'w') as f:
    json.dump(result_serializable_MLP2, f, indent=4)


print("Tranining of CNN3 is starting")
result_CNN3 = CNN_train_and_evaluate('cnn_3',
                                     model_CNN3, 
                                     train_loader, 
                                     test_loader, 
                                     learning_rate,
                                     num_epochs,
                                     device)
print("CNN_3 Training and evaluation finished")
PATH = './part3/results/CNN_3.pth'
torch.save(model_CNN3.state_dict(), PATH)

result_serializable_CNN3 = convert_to_serializable(result_CNN3)
# Save to JSON file
with open('./part3/results/result_CNN3.json', 'w') as f:
    json.dump(result_serializable_CNN3, f, indent=4)


print("Tranining of CNN4 is starting")
result_CNN4 = CNN_train_and_evaluate('cnn_4',
                                     model_CNN4, 
                                     train_loader, 
                                     test_loader, 
                                     learning_rate ,
                                     num_epochs,
                                     device)
print("CNN_4 Training and evaluation finished")
PATH = './part3/results/CNN_4.pth'
torch.save(model_CNN4.state_dict(), PATH)

result_serializable_CNN4 = convert_to_serializable(result_CNN4)
# Save to JSON file
with open('./part3/results/result_CNN4.json', 'w') as f:
    json.dump(result_serializable_CNN4, f, indent=4)


print("Tranining of CNN5 is starting")
result_CNN5 = CNN_train_and_evaluate('cnn_5',
                                     model_CNN5, 
                                     train_loader, 
                                     test_loader, 
                                     learning_rate,
                                     num_epochs,
                                     device)
print("CNN_5 Training and evaluation finished")
PATH = './part3/results/CNN_5.pth'
torch.save(model_CNN5.state_dict(), PATH)

result_serializable_CNN5 = convert_to_serializable(result_CNN5)
# Save to JSON file
with open('./part3/results/result_CNN5.json', 'w') as f:
    json.dump(result_serializable_CNN5, f, indent=4)