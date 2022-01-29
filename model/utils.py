import torch
import torch.nn as nn
def deeplist2module(layers):
    ret=[]
    for l in layers:
        if(type(l)==type([])):
            if(type(l[0])!=type([])):
                ret.append(nn.ModuleList(l))
            else:
                ret.append(deeplist2module(l))
    return nn.ModuleList(ret)

if __name__=='__main__':
    layers=[[nn.ReLU(),nn.ReLU()],[nn.ReLU()]]
    print(deeplist2module(layers))