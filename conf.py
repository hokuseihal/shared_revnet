import torch
from timm.scheduler.step_lr import StepLRScheduler
def get_optim_scheduler(key,model,epoch=None):
    if(key=='resnet'):
        optimizer=torch.optim.SGD(lr=0.1,momentum=0.9,weight_decay=1e-4,params=model.parameters())
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[epoch//3,2*epoch//3],gamma=0.1)
        return optimizer,scheduler
    if(key=='resnetcifar100'):

        optimizer=torch.optim.SGD(lr=0.1,momentum=0.9,weight_decay=5e-4,params=model.parameters())
        scheduler = StepLRScheduler(optimizer, decay_t=epoch//4, decay_rate=2e-1, warmup_t=400, warmup_lr_init=0)
        return optimizer,scheduler
    else:
        raise NotImplementedError

from timm.scheduler.step_lr import StepLRScheduler
from timm import create_model
if __name__=="__main__":
    model = create_model('resnet34')
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-1)
    scheduler=StepLRScheduler(optimizer,decay_t=10,decay_rate=2e-1,warmup_t=5,warmup_lr_init=0)
    for e in range(100):
        scheduler.step(e)
        print(e,scheduler.optimizer.param_groups[0]['lr'])