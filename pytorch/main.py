# This code is a derivative of one of the d2l linear regression examples. 
# THe purpose of this code is simply knowledge sharing 
# More information in README 


#Step 1 import libraries
import torch
from torch.utils import data
# `nn` is an abbreviation for neural networks
from torch import nn 

#Step 2: Create Dataset 
#Define a function to generate noisy data
def synthetic_data(m, c, num_examples):  
    """Generate y = mX + c + bias."""
    X = torch.normal(0, 1, (num_examples, len(m)))
    y = torch.matmul(X, m) + c
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_m = torch.tensor([2, -3.4])
true_c = 4.2
features, labels = synthetic_data(true_m, true_c, 1000)

print('features:', features[0],'\nlabel:', labels[0])



#Step 3: Read dataset and create small batch
#define a function to create a data iterator. Input is the features and labels from synthetic data
# Output is iterable batched data using torch.utils.data.DataLoader 
def load_array(data_arrays, batch_size, is_train=True):  
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

#Optional to see mini batches
#mini_batch = (iter(data_iter))
#for i in mini_batch:
#   print (i)
#   next(mini_batch)



#Step4: Define model & initialization
# Create single layer feed-forward network with 2 inputs and 1 outputs.
net = nn.Linear(2, 1)

#Initialize model params 
net.weight.data.normal_(0, 0.01)
net.bias.data.fill_(0)

#Step 5: Define loss function
# mean squared error loss function
loss = nn.MSELoss()

#Step 6: Define optimization algorithm 
# implements a stochastic gradient descent optimization method
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


# Step 7: Training 
# Use complete training data for n epochs, iteratively using a minibatch features and label 
# For each minibatch: 
#   Compute predictions by calling net(X) and calculate the loss l 
#   Calculate gradients by running the backpropagation
#   Update the model parameters using optimizer
#   Compute the loss after each epoch and print it to monitor progress
num_epochs = 5
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad() #sets gradients to zero
        l.backward() # back propagation
        trainer.step() # parameter update
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


#Results
m = net.weight.data
print('error in estimating m:', true_m - m.reshape(true_m.shape))
c = net.bias.data
print('error in estimating c:', true_c - c)


