import torch

NUM_STEPS=10000

def func(param, x):
    a4, a3, a2, a1, a0 = param
    p4 = a4 * x * x * x * x
    p3 = a3 * x * x * x
    p2 = a2 * x * x
    p1 = a1 * x
    p0 = a0
    return p4 + p3 + p2 + p1 + p0

def find_min_torch(param):
    x = torch.rand(1, requires_grad=True)
    opt = torch.optim.Adam([x], lr=0.01)
    last = None

    for s in range(NUM_STEPS):
        y = func(param, x)
        y.backward()
        opt.step()
        opt.zero_grad()
    print("x={}".format(x.item()))
    print("y={}".format(y.item()))

def find_min_no_optimizer(param):
    x = torch.rand(1, requires_grad=True)
    last = None

    for s in range(NUM_STEPS):
        y = func(param, x)
        y.backward()
        factor = torch.pow(1./torch.abs(x.grad.data.detach()), 1.1)
        if factor > 0.01:
            factor.data = torch.Tensor([0.01])
        factor = factor.squeeze(-1)
        x.data -= (factor * x.grad.data)
        x.grad.data.zero_()

    print("x={}".format(x.item()))
    print("y={}".format(y.item()))

if __name__ == "__main__":
    find_min_torch((4,1,3,2,1))
    find_min_no_optimizer((4,1,3,2,1))

    find_min_torch((40,1123,32,21,1))
    find_min_no_optimizer((40,1123,32,21,1))

