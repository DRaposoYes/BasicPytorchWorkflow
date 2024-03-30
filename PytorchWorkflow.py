import torch
from torch import nn # nn contains all building blocks for neural networks
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#check pytorch working
print("check pytorch working")
print(torch.__version__)


##1 data prepairing and loading

## create known parametars
print("create known parametars")
print(">")

weight = 0.7
bias = 0.3


## create 
print("create data")
print(">")

start = 0.2
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) 
y = weight * X + bias

print(X[:10], y[:10])
print(len(X), len(y))

## Spliting data into training and test sets
print("Spliting data into training and test sets")
print(">")

# create train/test split
print("create train/test split")
print(">")

train_split = int(0.8 * len(X))
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

    plt.scatter(test_data, test_labels, c="g", s=4, label="testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14});
    plt.show()



## 2. Build First Pytorch model, linear regression model

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
        return self.weights * x + self.bias # this is linear regression formula
    

print("create random seed")
print(">")

torch.manual_seed(42)

model_0 = LinearRegressionModel() 

print("show parameters of model")
print(">")

print(list(model_0.parameters()))

print("list named parameters of model")
print(">")

print(model_0.state_dict())

print("make random prediction using torch infence mode, make model predictions with model (random)")
print(">")

with torch.inference_mode():
    y_preds = model_0(X_test)
print(y_preds)




## 3. train model

print("train model")
print(">")

##loss function

print("setup loss function")
print(">")

loss_fn = nn.L1Loss()



##setup optimizer
print("setup optimizer")
print(">")
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)


## Building a training loop in Pytorch and testing loop
print("Building a training loop")
print(">")


torch.manual_seed(42)
epochs = 200

epoch_count = []
loss_values = []
test_loss_values = []


## step 0 loop trhough data and train

print(model_0.state_dict())

for epoch in range(epochs):

    model_0.train()

    ## 1 forward
    y_pred = model_0(X_train)

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
    model_0.eval()
    with torch.inference_mode():
        ## forward pass
        test_pred = model_0(X_test)
        ## Calculate loss
        test_loss = loss_fn(test_pred, y_test)
    if epoch % 10 == 0:

        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)

        print(f"epoch: {epoch} | loss: {loss} | Test loss: {test_loss} ")
        print(model_0.state_dict())



with torch.inference_mode():
    y_preds_new = model_0(X_test)

print(model_0.state_dict())



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


print("save and load model")
print(">")

## 1 create model directory

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

## 2 Create model save path
MODEL_NAME = "MyFirstModel_Model0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(MODEL_SAVE_PATH)

## 3 save mode state dict

print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)


print("loading model")
print(">")


## load model
loaded_model_0 = LinearRegressionModel()

loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

print("loaded model")
print(">")

print(loaded_model_0.state_dict())


print("check if loaded model is actual saved model")
print(">")

loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

print(loaded_model_preds)


print("it is the same")
print(">")

print(y_preds_new == loaded_model_preds)



print("see results of loaded model")
print(">")
##see results of loaded model
plot_predictions(predictions=loaded_model_preds);


print("end script")