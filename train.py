import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import MyTrainDataset, MyValandTestDataset, MyValandTestBlockDataset, MyDDataset
from model import RFNet,VRCNN,VRCNN_H,DenseNet,MyNet
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim 
from tensorboardX import SummaryWriter
import math

def train(args, model, device, train_loader, optimizer, epoch, writer, fold = 0):
    
    model.train()
    train_loss = 0
    for batch_idx, batch_data in enumerate(train_loader):
        data1 = batch_data['image1'].float() 
        target = batch_data['target'].float()  # [batch-size,channels,height,width] dtype=torch.float32

        data1, target = data1.to(device), target.to(device)

        optimizer.zero_grad()   
        output = model(data1)  

        loss = F.mse_loss(output,target)  
        train_loss += loss.item()

        loss.backward()   
        optimizer.step()  

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                args.epochs * fold + epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    train_loss /= len(train_loader)
    writer.add_scalar('Train/Loss', train_loss, args.epochs * fold + epoch)
