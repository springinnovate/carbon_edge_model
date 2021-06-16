# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
import math


# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x) + 0.35*(2*torch.rand(x.size())-1)

# Prepare the input tensor (x, x^2, x^3).
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)
print(p)
print(xx)
print(x)
print(x.unsqueeze(-1))
# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(1, 100),
    torch.nn.Sigmoid(),
    torch.nn.Linear(100, 100),
    torch.nn.Sigmoid(),
    torch.nn.Linear(100, 100),
    torch.nn.Sigmoid(),
    torch.nn.Linear(100, 1),
    torch.nn.Flatten(0, 1)
)
# model = torch.nn.Sequential(
#     torch.nn.Linear(3, 40),
#     torch.nn.Linear(40, 40),
#     torch.nn.Linear(40, 40),
#     torch.nn.Linear(40, 1),
#     torch.nn.Flatten(0, 1)
# )
loss_fn = torch.nn.MSELoss(reduction='sum')
#loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use RMSprop; the optim package contains many other
# optimization algorithms. The first argument to the RMSprop constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
last_loss = None

for t in range(20000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x.unsqueeze(-1))

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        if last_loss is not None:
            if loss.item() - last_loss > 0:
                learning_rate *= 0.95
            else:
                learning_rate *= 1.05
            loss_rate = (last_loss-loss.item())/last_loss
            if loss_rate < 0.001:
                break
            print(t, loss.item(), learning_rate, loss_rate)
        last_loss = loss.item()
        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(x, y_pred.detach().numpy())
        ax.grid()
        fig.savefig(f"test{t}.png")


    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()


#linear_layer = model[0]
#print(linear_layer.weight[:, 0].item())
#print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

