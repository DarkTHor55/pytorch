

import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

torch.empty(2,3)
# set vlaues which are already presnt in that location which is located to this tensor
# type is torch.Tensor

torch.zeros(2,3)
# set all default values as zero

torch.ones(2,3)
# set all default values as one

torch.rand(2,3)
# set all default values as random btween 0 1

#  we can descide random value also so, if someone try again values sould be same
torch.manual_seed(100)
torch.rand(2,3)

x=torch.tensor([[1,2,3],[4,5,6]])

# arange
print("using arange ->", torch.arange(0,10,2))

# using linspace
print("using linspace ->", torch.linspace(0,10,10))

# using eye(identify metrix) with 5*5
print("using eye ->", torch.eye(5))

# using full ( 3*3 with value 5 )
print("using full ->", torch.full((3, 3), 5))

#create a tensor with same shape with some other tensor but value will diffrent
torch.empty_like(x)

torch.zeros_like(x)

torch.ones_like(x)

# torch.rand_like(x)
# only sopportint by default
#due to float value it will not work

torch.rand_like(x,dtype=float)

x.dtype

torch.tensor([[1,2,3],[4,5,6]],dtype=torch.int32) # converting into int 32

torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float64) # converting into float

x.to(torch.float32) # converting into float 32

x=torch.empty(2,2)
x

x+x

x-1

x*100

x/100

y=torch.ones(2,3)
z=torch.ones(2,3)

y=y+z
y

y=y-z
y

y=y*z*2
y

y=(y*3)/(z*2)
y

torch.round(y)

torch.ceil(y)

e = torch.randint(size=(2,3), low=0, high=10, dtype=torch.float32)
e

# sum
torch.sum(e)
# sum along columns
torch.sum(e, dim=0)
# sum along rows
torch.sum(e, dim=1)

# mean
torch.mean(e)
# mean along col
torch.mean(e, dim=0)

# median
torch.median(e)

# max and min
torch.max(e)
torch.min(e)

# product
torch.prod(e)

# standard deviation
torch.std(e)

# variance
torch.var(e)

# argmax
torch.argmax(e)

# argmin
torch.argmin(e)

f = torch.randint(size=(2,3), low=0, high=10)
g = torch.randint(size=(3,2), low=0, high=10)

print(f)
print(g)

# matrix multiplcation
torch.matmul(f, g)

vector1 = torch.tensor([1, 2])
vector2 = torch.tensor([3, 4])

# transpose
torch.transpose(f, 0, 1)

h = torch.randint(size=(3,3), low=0, high=10, dtype=torch.float32)
h

# determinant
torch.det(h)

# inverse
torch.inverse(h)

i = torch.randint(size=(2,3), low=0, high=10)
j = torch.randint(size=(2,3), low=0, high=10)

print(i)
print(j)

# greater than
i > j
# less than
i < j
# equal to
i == j
# not equal to
i != j
# greater than equal to

# less than equal to

k = torch.randint(size=(2,3), low=0, high=10, dtype=torch.float32)
k

# log
torch.log(k)

# exp
torch.exp(k)

# sqrt
torch.sqrt(k)

# sigmoid
torch.sigmoid(k)

# softmax
torch.softmax(k, dim=0)

# relu
torch.relu(k)

m = torch.rand(2,3)
n = torch.rand(2,3)

print(m)
print(n)

m.add_(n)

device=torch.device('cuda')

torch.rand((2,3),device=device)

# cpu to gpu
x.to(device)

