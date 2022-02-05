import torch
from functools import partial
def get_optim_scheduler(key,model,epoch=None):
    if(key=='resnet'):
        optimizer=torch.optim.SGD(lr=0.1,momentum=0.9,weight_decay=1e-4,params=model.parameters())
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[epoch//3,2*epoch//3],gamma=0.1)
        return optimizer,scheduler
    else:
        raise NotImplementedError