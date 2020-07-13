import torch
from torchdiffeq import odeint
from pdb import set_trace as db
import matplotlib.pyplot as plt
import math

import torch.nn as nn

def func(t, y):

    return -y

class SecondOrderSolver:
    def __init__(self, func):
        self.func = func

    def solve(self, t, y0, dy0):
        # db()
        t0 = torch.Tensor([0])
        t = torch.cat([t0, t], dim=0)

        self.dy0 = dy0
        out = odeint(self.int_func, y0, t)
        return out[1]

    def int_func(self, t, _):
        # db()
        t0 = torch.Tensor([0])
        t = torch.Tensor([t]) + 1e-07  # 厳密に単調増加or単調減少である必要があるため
        t = torch.cat([t0, t], dim=0)
        out = odeint(self.func, self.dy0, t)
        return out[1]


class Oscillation:
    def __init__(self, k):
        self.mat = torch.Tensor([[0, 1],
                                 [-k, 0]])

    def solve(self, t, x0, dx0):
        y0 = torch.cat([x0, dx0])
        out = odeint(self.func, y0, t)
        return out

    def func(self, t, y):
        # print(t)
        out = y @ self.mat
        return out



if __name__=="__main__":

    # y0 = torch.Tensor([0]).requires_grad_()
    x0 = torch.Tensor([1])
    dx0 = torch.Tensor([0])


    # t = torch.Tensor([0, 2]) # t0 = t[0]となる
    import numpy as np
    t = torch.linspace(0, 4 * np.pi, 1000)

    # solver = SecondOrderSolver(func)
    solver = Oscillation(1)
    out = solver.solve(t, x0, dx0)
    # print(out)


    plt.plot(t, out[:, 0], label='x: pos')
    plt.plot(t, out[:, 1], label='dx/dt: vel')
    # plt.axes().set_aspect('equal', 'datalim')  # 縦横比を1:1にする
    plt.grid()
    plt.legend()
    # plt.xlim(0,2)
    plt.show()