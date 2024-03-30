import torch
from torch import nn # nn contains all building blocks for neural networks
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


print("create known parametars")
print(">")

weight = 0.8
bias = 0.2

print("create data")
print(">")

start = 1
end = 0.2
step = -0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) 
y = weight * X - bias

print(X[:10], y[:10])
print(len(X), len(y))

print("Spliting data into training and test sets")
print(">")

train_split = int(0.80 * len(X))
print(train_split)

X_train, y_train = X[:train_split], y[:train_split]

X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))


print("create plot vizualization")
print(">")
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    ""
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c="b", s=4, label="training data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14});
    plt.show()


print("create linear regression model class")
print(">")

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1,
                                            requires_grad=True,
                                            dtype=torch.float))
        #forward method to define computation in model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x - self.bias # this is linear regression formula


## ask user for lucky number for manual seed
random_number = input("whats your lucky number?: ")

torch.manual_seed(random_number)
Model_DownwardsLine = LinearRegressionModel() 




print("train model")
print(">")

print("setup loss function")
print(">")

loss_fn = nn.L1Loss()


##setup optimizer
print("setup optimizer")
print(">")
optimizer = torch.optim.SGD(params=Model_DownwardsLine.parameters(),
                            lr=0.001)


## Building a training loop in Pytorch and testing loop
print("Building a training loop")
print(">")

torch.manual_seed(random_number)


epochs = int(input("how many times do you want to train the model for?:"))

epoch_count = []
loss_values = []
test_loss_values = []


## looping: Enable training mode, forward (connecting line between neurons), calculate loss, setup optimizer, perform backpropegation (model "Reflects on itself"), optimizer step, 
## test model. Repeat loop per epoch
for epoch in range(epochs):

    Model_DownwardsLine.train()

    ## 1 forward
    y_pred = Model_DownwardsLine(X_train)

    ##2 whats the loss

    loss = loss_fn(y_pred, y_train)
    print(f"Loss {loss}")

    ##3 optimizer zero grad

    optimizer.zero_grad()

    ## 4 perform backpropagation on the loss with respect to the parameters of the model
     
    loss.backward()
    
    ##5 perform gradient descent
    optimizer.step()

    ##testing loop
    Model_DownwardsLine.eval()
    with torch.inference_mode():
        ## forward pass
        test_pred = Model_DownwardsLine(X_test)
        ## Calculate loss
        test_loss = loss_fn(test_pred, y_test)
    if epoch % 10 == 0:

        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)

        print(f"epoch: {epoch} | loss: {loss} | Test loss: {test_loss} ")
        print(Model_DownwardsLine.state_dict())



## make prediction based on model training
with torch.inference_mode():
    y_preds_new = Model_DownwardsLine(X_test)



## Visualize training curve

print("see training curve of model")
print(">")

plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();
plt.show()


##see results of model
print("see results of model")
print(">")

plot_predictions(predictions=y_preds_new);