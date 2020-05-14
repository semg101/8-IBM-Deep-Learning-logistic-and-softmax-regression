#Preparation---------------------------------------------------------------
# Import the libraries we need for this lab

# Using the following line code to install the torchvision library
# !conda install -y torchvision
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np

# Display data

def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(28, 28), cmap='gray')
    plt.title('y = ' + str(data_sample[1].item()))

#Make Some Data----------------------------------------------------------------------------
#Load the training dataset by setting the parameters train to True and convert it to a tensor by placing a transform object in the argument transform.
# Create and print the training dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
print("Print the training dataset:\n ", train_dataset)

#Load the testing dataset by setting the parameters train to False and convert it to a tensor by placing a transform object in the argument transform.
# Create and print the validating dataset
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
print("Print the validating dataset:\n ", validation_dataset)

# Print the type of the element
print("Type of data element: ", train_dataset[0][1].type())

#Print out the fourth label:
# Print the label
print("The label: ", train_dataset[3][1])

#Plot the the fourth sample:
# Plot the image
print("The image: ", show_data(train_dataset[3]))

#You see its a 1. Now, plot the third sample:
# Plot the image
show_data(train_dataset[2])

#The Softmax function requires vector inputs. If you see the vector shape, you'll note it's 28x28.
# Print the shape of the first element in train_dataset
train_dataset[0][0].shape

#Practice------------------------------------------------------------------------------
# Set input size and output size
input_dim = 28 * 28
output_dim = 10

# Practice: Create a softmax classifier by using sequenital
model = nn.Sequential(nn.Linear(input_dim, output_dim))

#Define the Softmax Classifier, Criterion function, Optimizer, and Train the Model---------------------------------------------
#View the size of the model parameters:
# Print the parameters
print('W: ', list(model.parameters())[0].size())
print('b: ', list(model.parameters())[1].size())

#Define the dataset loader:
# Define the learning rate, optimizer, criterion and data loader
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)
Train the model and determine validation accuracy:

# Train the model
n_epochs = 10
loss_list = []
accuracy_list = []
N_test = len(validation_dataset)

​

def train_model(n_epochs):

    for epoch in range(n_epochs):

        for x, y in train_loader:

            optimizer.zero_grad()

            z = model(x.view(-1, 28 * 28))

            loss = criterion(z, y)

            loss.backward()

            optimizer.step()

        correct = 0

        

        #perform a prediction on the validation  data  
        for x_test, y_test in validation_loader:
            z = model(x_test.view(-1, 28 * 28))
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        loss_list.append(loss.data)
        accuracy_list.append(accuracy)

train_model(n_epochs)

#Analyze Results--------------------------------------------------------
#Plot the loss and accuracy on the validation data:
# Plot the loss and accuracy
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(loss_list, color = color)
ax1.set_xlabel('epoch', color = color)
ax1.set_ylabel('total loss', color = color)
ax1.tick_params(axis = 'y', color = color)
    
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('accuracy', color = color)  
ax2.plot( accuracy_list, color = color)
ax2.tick_params(axis = 'y', color = color)
fig.tight_layout()

#Plot the first five misclassified samples:
# Plot the misclassified samples
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    if yhat != y:
        show_data((x, y))
        plt.show()
        print("yhat: ",yhat)
        count += 1
    if count >= 5:
        break