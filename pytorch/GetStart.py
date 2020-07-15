from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)
x = torch.rand(5, 3)
print(x)
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
x = torch.tensor([5.5, 3])
print(x)
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)
print(x.size())
y = torch.rand(5, 3)
print("**********************")
print(x + y)
result = torch.empty(5, 3)
torch.add(x, y, out=result)

print("**********************")
print(result)

print("**********************")
y.add_(x)
print(y)

print("**********************")
print(x)
print(x[:, 1])

print("**********************")
x = torch.randn(4, 4)
print(x)
y = x.view(16)
print(y)
z = x.view(-1, 8)
print(z)
z = x.view(8, -1)
print(z)
print(x.size(), y.size(), z.size())

print("**********************")
x = torch.randn(1)
print(x)
print(x.item())
x = torch.randn(3, 2)
print(x)
print(x[1, 1].item())

print("**********************")
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

print("**********************")
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

print("**********************")
x = torch.randn(1)
print(x)
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    # x = x.to(device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))


