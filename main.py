from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR100 as Dataset

from conf import get_optim_scheduler
from model import modeldic


def operate(phase):
    mloss = []
    macc = []
    log = {}
    with torch.set_grad_enabled(phase == 'train'):

        if phase == 'train':
            model.train()
            loader = trainloader
        else:
            model.eval()
            loader = valloader
        for idx, (data, target) in enumerate(loader):

            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            acc = (output.argmax(1) == target).float().mean()
            mloss.append(loss.item())
            macc.append(acc.item())
            log.update({f'{phase}/loss': loss.item(), f'{phase}/acc': acc.item()})
            if (phase == 'train'):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if (scheduler is not None):
                    scheduler.step(epoch=e * len(trainloader) + idx)
                    log.update({'lr': optimizer.param_groups[0]['lr']})

                if (args.wandb): wandb.log(log)
            print(f"{e=},{idx}/{len(loader)},{loss=:2.4f},{acc=:2.4f},lr={optimizer.param_groups[0]['lr']:.3e},{phase}")

        mloss = np.mean(mloss)
        macc = np.mean(macc)
        print(f"{e},LOSS:{mloss},ACC:{macc}")
        if (args.wandb): wandb.log({f'{phase}/mloss': mloss, f'{phase}/macc': macc})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=128, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', default='revnet18_r2')
    parser.add_argument('--paramshare', default=1, type=int)
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--tag', default='')
    parser.add_argument('--setting', required=True)
    args = parser.parse_args()
    if (args.wandb): import wandb

    device = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()

    model = modeldic[args.model](num_classes=100, parameter_share=args.paramshare).to(device)
    optimizer, scheduler, train_transform, val_transform = get_optim_scheduler(args.setting, model,
                                                                               args.epoch * 50000 // args.batchsize)
    trainloader = torch.utils.data.DataLoader(
        Dataset(train=True, download=True, transform=train_transform, root='../data/cifar100'),
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=cpu_count())
    valloader = torch.utils.data.DataLoader(
        Dataset(train=False, download=True, transform=val_transform, root='../data/cifar100'),
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=cpu_count())

    if (args.wandb): wandb.init(project='shared_revnet', name=args.model + args.tag,
                                config={"optimizer": str(optimizer), "scheduler": str(scheduler), "model": args.model,
                                        "batchsize": args.batchsize, "train_transform": str(train_transform),
                                        'val_transform': str(val_transform), "setting": args.setting})
    for e in range(args.epoch):
        operate('train')
        operate('val')

    if (args.wandb): wandb.finish()
