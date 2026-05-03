import torch
import torch.nn as nn

model = nn.Linear(1,1)

x = torch.tensor([[1.0],[2.0],[3.0]])
y = torch.tensor([[2.0],[4.0],[6.0]])

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(100):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Prediction for 4:", model(torch.tensor([[4.0]])))
