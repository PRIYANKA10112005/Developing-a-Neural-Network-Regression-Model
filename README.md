# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:

### Register Number:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("/content/Exp-1.csv")
df

x = df[["Input"]].values
y = df[["Output"]].values
xt,xst,yt,yst = train_test_split(x,y,test_size=0.2,random_state=42)

scale1 = MinMaxScaler()
scale2=MinMaxScaler()
xt = scale1.fit_transform(xt)
xst = scale2.fit_transform(xst)


xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)
yt = torch.FloatTensor(yt)
yst = torch.FloatTensor(yst)

class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
    def forward(self,x):
        return self.network(x)

# Initialize the Model, Loss Function, and Optimizer
model = neuralnet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# Train the model
epochs = 1000
losses=[]
for i in range(epochs):
    optimizer.zero_grad()
    pred = model(xt)
    loss = criterion(pred, yt)
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"{i}/{epochs} Loss: {loss.item():.4f}")
        losses.append(loss.item())

# Tesing for new input
new = scale1.transform([[16]])
new = torch.FloatTensor(new)

pred = model(new)
print(pred.item())

# Evaluating loss for testing data
with torch.no_grad():
    pred=model(xst)
    loss_test=criterion(pred,yst)
    print(loss_test)

# Plot the loss curve

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
```

### Dataset Information
<img width="182" height="432" alt="image" src="https://github.com/user-attachments/assets/a06098b9-594e-42d2-9b58-58355fa8fc72" />


### OUTPUT
<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/9d9d4a11-0149-4a66-b364-7333d1b933f8" />

### Training Loss Vs Iteration Plot
Include your plot here

### New Sample Data Prediction
Include your sample input and output here

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
