from __future__ import print_function
import torch
import numpy

# x = torch.rand(5, 3)
# print(x)
#
# print(torch.cuda.is_available())
# # torch.set_default_dtype()
# # torch.get_default_dtype()
# print(torch.tensor([1.2, 3]).dtype)
# print(torch.get_default_dtype())
# torch.set_default_dtype(torch.float64)
# print(torch.tensor([1.2, 3]).dtype)
# print(torch.get_default_dtype())
#
# # torch.set_default_tensor_type(t)
# torch.set_default_dtype(torch.float32)
# print(torch.tensor([1.2, 3]).dtype)
# torch.set_default_tensor_type(torch.DoubleTensor)
# print(torch.tensor([1.2, 3]).dtype)
#
# # torch.numel(input)
#
# a = torch.randn(1, 2, 3, 4, 5)
# print(torch.numel(a))
# a = torch.zeros(4, 4)
# print(torch.numel(a))
#
# # torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)
#
# # torch.set_flush_denormal(mode) → bool
# #
# print(torch.set_flush_denormal(True))
# print(torch.tensor([1e-323], dtype=torch.float64))
# print(torch.set_flush_denormal(False))
# print(torch.tensor([1e-323], dtype=torch.float64))
#
# # torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False) → Tensor
# a = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
# print(a)
# print(torch.tensor([0, 1]))
# print(torch.tensor([[0.11111, 0.2222, 0.3333]], dtype=torch.float64, device=torch.device('cuda:0')))
# print(torch.tensor(3.1415))
# print(torch.tensor([]))
#
# # torch.sparse_coo_tensor(indices, values, size=None, dtype=None, device=None, requires_grad=False) → Tensor
# i = torch.tensor([[0, 1, 1], [2, 0, 2]])
# v = torch.tensor([3, 4, 5], dtype=torch.float32)
# print(torch.sparse_coo_tensor(i, v, [2, 4]))
# print(torch.sparse_coo_tensor(i, v))
#
# # torch.as_tensor(data, dtype=None, device=None) → Tensor
# a = numpy.array([1, 2, 3])
# t = torch.as_tensor(a)
# print(a)
# print(t)
# t[0] = -1
# print(a)
# print(t)
#
# a = numpy.array([1, 2, 3])
# t = torch.as_tensor(a, device=torch.device('cuda'))
# print(a)
# print(t)
# t[0] = -1
# print(a)
# print(t)
#
# # torch.from_numpy(ndarray) → Tensor
# a = numpy.array([1, 2, 3])
# t = torch.from_numpy(a)
# print("torch.from_numpy(ndarray) → Tensor")
# print(a)
# print(t)
# t[0] = -1
# print(a)
# print(t)
#
# # torch.as_strided(input, size, stride, storage_offset=0) → Tensor
# x = torch.randn(3, 3)
# print(x)
# t = torch.as_strided(x, (2, 2), (1, 2))
# print(t)
# t = torch.as_strided(x, (2, 2), (1, 2), 1)
# print(t)
# t = torch.as_strided(x, (3, 2), (2, 2))
# print(t)
#
# # torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
# print(torch.zeros(2, 3, 4, 5))
# print(torch.zeros(5))
#
# # torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False,
# # memory_format=torch.preserve_format) → Tensor
#
# input = torch.empty(2, 3)
# print(torch.zeros_like(input))
#
# # torch.ones(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
# print(torch.ones(2, 3))
# print(torch.ones(5))
#
# # torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) →
# # Tensor
# print(torch.arange(5))
# print(torch.arange(1, 4))
# print(torch.arange(1, 2.5, 0.5))
#
# # torch.linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# # → Tensor
# print(torch.linspace(3, 10, steps=5))
# print(torch.linspace(-10, 10))
#
# # torch.logspace(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None,
# # requires_grad=False) → Tensor
# print(torch.logspace(-10, 10, 5))
# print(torch.logspace(-1, 1, 5))
# print(torch.logspace(start=0.1, end=1.0, steps=1))
# print(torch.logspace(start=2, end=2, steps=1, base=2))
#
# # torch.eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
# a = torch.eye(3)
# print(torch.eye(3))
#
# # torch.empty(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
# # → Tensor
# print(torch.empty(3, 2))
#
# # torch.empty_like(input, dtype=None, layout=None, device=None, requires_grad=False,
# # memory_format=torch.preserve_format) → Tensor
# print(torch.empty_like(a))
#
# # torch.empty_strided(size, stride, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) →
# # Tensor
#
# # torch.full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
# print(torch.full((2, 3), 3.141592))
# print(torch.full_like(a, 111))
#
#
# torch.set_default_dtype(torch.float32)
# # torch.quantize_per_tensor(input, scale, zero_point, dtype) → Tensor
# a = torch.tensor([-1.2, 0.0, 1.0, 2.0], dtype=torch.float64)
# print(a)
# b = torch.quantize_per_tensor(a, 0.1, 10, torch.quint8)
# print(b)
#
# x = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
# print(torch.quantize_per_channel(x, torch.tensor([0.1, 0.01]), torch.tensor([10, 0]), 0, torch.quint8))

#########################
# torch.cat(tensors, dim=0, out=None) → Tensor
a = torch.randn(3, 2)
print(a)
print(torch.cat((a, a, a), 0))
print(torch.cat((a, a, a), 1))
print(torch.cat((a, a, a), -2))

