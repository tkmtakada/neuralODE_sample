import torch
from torchdiffeq import odeint
from pdb import set_trace as db
import torch.nn as nn
def func(t, y):

    return t
# =========================
# MNISTで試す
# =========================
class ODENet(nn.Module):
    def __init__(self):
        pass
    def forward(self, x, t):
        # =--------------
        #
        # =--------------

        return x


class ODEfunc(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ODEfunc, self).__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(32, out_dim)
        self.tanh = nn.Tanh()
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        print(x)
        print(self.integration_time)
        out = odeint(self.odefunc, x, self.integration_time) #, rtol=args.tol, atol=args.tol)
        return out[1]  # out[0]には初期値が入っているので．

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


if __name__=="__main__":
    f = ODEfunc(2,2)
    model = ODEBlock(f)

    y0 = torch.Tensor([[0, 0]]).requires_grad_()
    t = torch.Tensor([0, 1]) # t0 = t[0]となる

    # out = odeint(func, y0, t)
    out = model(y0)
    print(out)
    l = torch.sum(out)
    l.backward()
    db()
