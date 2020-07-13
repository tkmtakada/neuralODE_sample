import torch
from torchdiffeq import odeint
from pdb import set_trace as db
import matplotlib.pyplot as plt
import torch.nn as nn

def func(t, y):

    return t

if __name__=="__main__":

    # y0 = torch.Tensor([0]).requires_grad_()
    y0 = torch.Tensor([0])


    # t = torch.Tensor([0, 1]) # t0 = t[0]となる
    t = torch.linspace(0,2,100)

    out = odeint(func, y0, t)
    print(out)
    # l = torch.sum(out)
    # l.backward()

    # draw graph

    plt.plot(t, out)
    plt.axes().set_aspect('equal', 'datalim')  # 縦横比を1:1にする
    plt.grid()
    plt.xlim(0,2)
    plt.show()