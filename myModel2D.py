import torch
import torch.nn as nn
from myResNet2D import resnet18, resnet34
from myAE2D import Baseline

class MyRsCliModel(nn.Module):
    def __init__(self, net1, indim=5, outdim=2):
        super().__init__()
        self.net1 = net1

        self.sigmoid = nn.Sigmoid()

        self.indim = indim
        self.outdim = outdim
        self.linear1 = nn.Linear(in_features=self.indim, out_features=16)
        self.linear2 = nn.Linear(in_features=16, out_features=16)
        self.linear3 = nn.Linear(in_features=16, out_features=self.outdim)
        self.relu = nn.ReLU()

        self.linear4 = nn.Linear(in_features=4, out_features=2)



    def forward(self, x, xcli):


        x1, xf = self.net1(x)

        xcli = self.linear1(xcli)
        xcli = self.relu(xcli)
        xcli = self.linear2(xcli)
        xcli = self.relu(xcli)
        xcli = self.linear3(xcli)

        xrscli = torch.cat([x1, xcli], dim=1)
        xrscli = self.linear4(xrscli)

        return xrscli, xf

class MyRsCliReconst(nn.Module):
    def __init__(self, net1, indim=4, outdim=2):
        super().__init__()
        self.net1 = net1

        self.sigmoid = nn.Sigmoid()

        self.indim = indim
        self.outdim = outdim
        self.linear1 = nn.Linear(in_features=self.indim, out_features=16)
        self.linear2 = nn.Linear(in_features=16, out_features=16)
        self.linear3 = nn.Linear(in_features=16, out_features=self.outdim)
        self.relu = nn.ReLU()
        self.linear4 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x, xcli):

        recx, x1, xf = self.net1(x)

        xcli = self.linear1(xcli)
        xcli = self.relu(xcli)
        xcli = self.linear2(xcli)
        xcli = self.relu(xcli)
        xcli = self.linear3(xcli)

        xrscli = torch.cat([x1, xcli], dim=1)
        xrscli = self.linear4(xrscli)

        return recx, xrscli, xf

class MyRsCliDisent(nn.Module):
    def __init__(self, net1, indim=5, outdim=2):
        super().__init__()
        self.net1 = net1

        self.sigmoid = nn.Sigmoid()

        self.indim = indim
        self.outdim = outdim
        self.linear1 = nn.Linear(in_features=self.indim, out_features=16)
        self.linear2 = nn.Linear(in_features=16, out_features=16)
        self.linear3 = nn.Linear(in_features=16, out_features=self.outdim)
        self.relu = nn.ReLU()

        self.linear4 = nn.Linear(in_features=4, out_features=2)



    def forward(self, x, xcli):


        recx1, x1, x1dom, xae, xae2 = self.net1(x)

        xcli = self.linear1(xcli)
        xcli = self.relu(xcli)
        xcli = self.linear2(xcli)
        xcli = self.relu(xcli)
        xcli = self.linear3(xcli)

        xrscli = torch.cat([x1, xcli], dim=1)
        xrscli = self.linear4(xrscli)

        return recx1, xrscli, x1dom, xae, xae2

class DGRsCliModel(nn.Module):
    def __init__(self, net1, indim=5, outdim=2):
        super().__init__()
        self.net1 = net1

        self.sigmoid = nn.Sigmoid()

        self.indim = indim
        self.outdim = outdim
        self.linear1 = nn.Linear(in_features=self.indim, out_features=16)
        self.linear2 = nn.Linear(in_features=16, out_features=16)
        self.linear3 = nn.Linear(in_features=16, out_features=self.outdim)
        self.relu = nn.ReLU()

        self.linear4 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x, xcli):

        x1, sfs = self.net1(x)

        xcli = self.linear1(xcli)
        xcli = self.relu(xcli)
        xcli = self.linear2(xcli)
        xcli = self.relu(xcli)
        xcli = self.linear3(xcli)

        xrscli = torch.cat([x1, xcli], dim=1)
        xrscli = self.linear4(xrscli)

        return xrscli, sfs

class DGBaselineRsCliModel(nn.Module):
    def __init__(self, net1, indim=4, outdim=2):
        super().__init__()
        self.net1 = net1

        self.sigmoid = nn.Sigmoid()

        self.indim = indim
        self.outdim = outdim
        self.linear1 = nn.Linear(in_features=self.indim+1, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=self.outdim)
        self.relu = nn.ReLU()

    def forward(self, x, xcli):

        x1, sfs1 = self.net1(x)

        x1 = torch.softmax(x1, dim=1)
        xrscli = torch.cat([x1[:, 1].view(x1.shape[0], -1), xcli], dim=1)
        xrscli = self.linear1(xrscli)
        xrscli = self.relu(xrscli)
        xrscli = self.linear2(xrscli)
        xrscli = self.relu(xrscli)
        xrscli = self.linear3(xrscli)


        return xrscli, sfs1

class DGdomRsCliModel(nn.Module):
    def __init__(self, net1, indim=4, outdim=2):
        super().__init__()
        self.net1 = net1

        self.sigmoid = nn.Sigmoid()

        self.indim = indim
        self.outdim = outdim
        self.linear1 = nn.Linear(in_features=self.indim+1, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=self.outdim)
        self.relu = nn.ReLU()

    def forward(self, x, xcli):

        x1, sfs1, clas2, sfs2, clast, f1, f11 = self.net1(x)

        x1 = torch.softmax(x1, dim=1)
        xrscli = torch.cat([x1[:, 1].view(x1.shape[0], -1), xcli], dim=1)
        xrscli = self.linear1(xrscli)
        xrscli = self.relu(xrscli)
        xrscli = self.linear2(xrscli)
        xrscli = self.relu(xrscli)
        xrscli = self.linear3(xrscli)


        return xrscli, sfs1, clas2, sfs2, clast, f1, f11

class DGaedomRsCliModel(nn.Module):
    def __init__(self, net1, indim=4, outdim=2):
        super().__init__()
        self.net1 = net1

        self.sigmoid = nn.Sigmoid()

        self.indim = indim
        self.outdim = outdim
        self.linear1 = nn.Linear(in_features=self.indim+1, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=self.outdim)
        self.relu = nn.ReLU()

    def forward(self, x, xcli):

        x1, sfs1, clas2, sfs2, clast, f1, f11, rec = self.net1(x)

        x1 = torch.softmax(x1, dim=1)
        xrscli = torch.cat([x1[:, 1].view(x1.shape[0], -1), xcli], dim=1)
        xrscli = self.linear1(xrscli)
        xrscli = self.relu(xrscli)
        xrscli = self.linear2(xrscli)
        xrscli = self.relu(xrscli)
        xrscli = self.linear3(xrscli)


        return xrscli, sfs1, clas2, sfs2, clast, f1, f11, rec

class DGaedomRsCliModelnodisent(nn.Module):
    def __init__(self, net1, indim=4, outdim=2):
        super().__init__()
        self.net1 = net1

        self.sigmoid = nn.Sigmoid()

        self.indim = indim
        self.outdim = outdim
        self.linear1 = nn.Linear(in_features=self.indim+1, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=self.outdim)
        self.relu = nn.ReLU()

    def forward(self, x, xcli):

        x1, rec = self.net1(x)
        x1 = torch.softmax(x1, dim=1)
        xrscli = torch.cat([x1[:, 1].view(x1.shape[0], -1), xcli], dim=1)
        xrscli = self.linear1(xrscli)
        xrscli = self.relu(xrscli)
        xrscli = self.linear2(xrscli)
        xrscli = self.relu(xrscli)
        xrscli = self.linear3(xrscli)

        return xrscli, rec

class CliModel(nn.Module):
    def __init__(self, indim=4, outdim=2):
        super().__init__()
        self.indim = indim
        self.outdim = outdim
        self.linear1 = nn.Linear(in_features=self.indim, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=self.outdim)
        self.relu = nn.ReLU()

    def forward(self, xcli):
        xrscli = xcli
        xrscli = self.linear1(xrscli)
        xrscli = self.relu(xrscli)
        xrscli = self.linear2(xrscli)
        xrscli = self.relu(xrscli)
        xrscli = self.linear3(xrscli)

        return xrscli

if __name__ == '__main__':
    print('*************************')
    model = MyRsCliModel(Baseline())
    print(model(torch.ones([4,3,64,64]), torch.ones([4, 5])))
