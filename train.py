
# In[]
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import argparse
import torch.nn.functional as F
import pdb
import tqdm

# In[]
class ODEnet(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ODEnet, self).__init__()

        odefunc = ODEfunc(dim=mid_dim)

        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm1d(mid_dim)
        self.ode_block = ODEBlock(odefunc)
        self.norm2 = nn.BatchNorm1d(mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        out = self.fc1(x)
        out = self.relu1(out)
        out = self.norm1(out)
        out = self.ode_block(out)
        out = self.norm2(out)
        out = self.fc2(out)

        # CrossEntropyの中にSoftmaxは含まれているらしい
        # out= self.softmax(out, dim=1)

        return out


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.seq = nn.Sequential(nn.Linear(dim, 124),
                                 nn.ReLU(),
                                 nn.Linear(124, 124),
                                 nn.ReLU(),
                                 nn.Linear(124, dim),
                                 nn.Tanh())

    def forward(self, t, x):
        out = self.seq(x)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time) #, rtol=args.tol, atol=args.tol)
        return out[1]  # out[0]には初期値が入っているので．


# In[]
def train(model):
    n_epoch = 20
    print("device : ", device)
    #device = 'cpu'
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    test_loss_per_epoch = []
    test_acc_per_epoch = []
    for epoch in range(n_epoch):
        train_loss_all = 0
        test_loss_all = 0
        acc = 0
        model.train()
        counter = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_all += loss.data
            acc += torch.sum(labels == torch.argmax(outputs,dim=1)).cpu().numpy()
        train_loss_all = train_loss_all/float(len(trainset))
        train_loss_per_epoch.append(train_loss_all)
        acc = acc/float(len(trainset))
        train_acc_per_epoch.append(acc)
        print("train: epoch: {:.4g}, loss: {:.4g}".format(epoch, train_loss_all, acc))

        # test
        model.eval()
        acc = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss_all += loss.data
            acc += torch.sum(labels == torch.argmax(outputs, dim=1)).cpu().numpy()
        test_loss_all = test_loss_all/float(len(trainset))
        test_loss_per_epoch.append(test_loss_all)
        acc = acc/float(len(testset))
        test_acc_per_epoch.append(acc)
        print("test: epoch: {}, loss: {:.4g}, acc : {:.4g}".format(epoch,test_loss_all, acc))
        if epoch%10 == 9:
            torch.save(model.state_dict(), "./weight_wpoch{}.pth".format(epoch))
    return (train_loss_per_epoch, test_loss_per_epoch), (train_acc_per_epoch, test_acc_per_epoch)


def loss_plot(train_loss, test_loss):
    epochs = np.arange(len(train_loss))
    plt.clf()
    plt.plot(epochs, train_loss, label='train')
    plt.plot(epochs, test_loss, label="test")
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig("./loss_{}.png".format(method))
    # plt.show()
def acc_plot(train_acc, test_acc):
    epochs = np.arange(len(train_acc))
    plt.clf()
    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, test_acc, label='test')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig("./accuracy_{}.png".format(method))
    # plt.show()

# In[]
def imshow(img):# img is assumed as having shape of channel, height, and width
    img = img / 2 + 0.5
    npimg = img.numpy()
    print(npimg.shape)
    #plt.imshow(np.transpose(npimg, (0,1,2)))
    plt.imshow(npimg,cmap='gray')

# In[]
if __name__=='__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--adjoint", action='store_true', help='use adjoint method')
    args = parser.parse_args()

    global method
    method = "adjoint" if args.adjoint else ""

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5, ))])

    trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root="./data",train=False,download=True,transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,shuffle=False,num_workers=2)
    classes = ('0','1','2', '3', '4','5','6','7','8','9')
    for images, labels in trainloader:
        print(images.shape)
        img = images[0]
        imshow(img.squeeze())
        break
    #model = Net(img_w = 28)
    # model = MLP(batch_size=100)
    model = ODEnet(in_dim=28*28, mid_dim=32, out_dim=10)
    loss, acc = train(model)
    loss_plot(loss[0], loss[1])
    acc_plot(acc[0], acc[1])
