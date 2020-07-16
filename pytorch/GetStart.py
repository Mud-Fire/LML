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

print("=============AUTOGRAD============")
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(z, out)
out.backward()
print(x.grad)

print("=============AUTOGRAD============")
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

print("=============AUTOGRAD============")
x = torch.randn(3, requires_grad=True)
y = x * 2
print(x)
print(y)
print(y.data.norm())
while y.data.norm() < 1000:
    y = y * 2
print(y)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())