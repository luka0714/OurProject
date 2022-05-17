import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn
from model import MyNet,VRCNN,RFNet,SRCNN,NewRFNet,Dianet,SRCNN_rl,VRCNN_rl,RFNet_rl,RFNet_v1,RFNet_v2,DeformConvNet,DeformConvNet_v2
from test import cal_psnr,cal_ssim
from PIL import Image
from utils import createFrame


from train import train
from test import test
from torch.utils.data import Dataset, DataLoader
from dataset import MyblockDataset,MyDataset
from tensorboardX import SummaryWriter

def generateDataset(length, fold, Dlist, train_path, val_path):
    val_start = fold*length
    val_end = val_start + length
    for m in range(val_start, val_end):
        val_path.append(Dlist[m])
    for n in range(0, val_start):
        train_path.append(Dlist[n])
    for p in range(val_end, len(Dlist)):
        train_path.append(Dlist[p])
    return train_path, val_path

def main():
    # Training settings

    writer = SummaryWriter()

    parser = argparse.ArgumentParser(description='Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 5)')

    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)') 
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train (default: 6)')  
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 1.0)')    #  w^1=w^0-lr*dw  b^1=b^0-lr*db
    parser.add_argument('--gamma', type=float, default=0.3, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')  
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')  
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')   
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')  
    parser.add_argument('--load-model', default=False,
                        help='For load the prepared Model')
    parser.add_argument('--train_input_file', default='Located_train_bubbleblowing_32.npy',
                        help='numpy file for training input dataset and my_size=?')
    parser.add_argument('--train_label_file', default='Located_target_bubbleblowing_32.npy',
                        help='numpy file for training label dataset and my_size=?')
    parser.add_argument('--test_file', default='txtfile/test_bubbleblowing.txt',
                        help='txt file for test dataset')
    parser.add_argument('--flag', type=bool, default=False, help='is output feature maps') 
    parser.add_argument('--outdir', type=str, default='output\\DEMO', help='output feature maps dir') 
    parser.add_argument('--cross_flag', type=bool, default=False, help='is use cross validation') 
    parser.add_argument('--k', type=int, default=10, help='Cross-Validation') 
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()   

    torch.manual_seed(args.seed)  

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,   
                       'pin_memory': True,
                       'shuffle': True}   

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # test set
    '''
    test_dataset = MyDataset(txt = 'data/test.txt')  
    test_loader = DataLoader(test_dataset,**test_kwargs)
    '''
    # args.load_model = True
    if(args.load_model):
        
        checkpoint = torch.load('model/VRCNN.pth')
        model = checkpoint['net']
        optimizer = checkpoint['optimizer']
        epochs = checkpoint['epoch'] + 1
        # test(model,device,test_loader,1,writer)

        # 生成模型预测帧
        for frame in range(2, 120):

            if frame < 9:
                data_path = "BQMall_100/frame_00" + str(frame) + ".png"
                target_path = "BQMall_100/frame_00" + str(frame + 1) + ".png"
            elif frame == 9:
                data_path = "BQMall_100/frame_00" + str(frame) + ".png"
                target_path = "BQMall_100/frame_0" + str(frame + 1) + ".png"
            elif frame < 99:
                data_path = "BQMall_100/frame_0" + str(frame) + ".png"
                target_path = "BQMall_100/frame_0" + str(frame + 1) + ".png"
            elif frame == 99:
                data_path = "BQMall_100/frame_0" + str(frame) + ".png"
                target_path = "BQMall_100/frame_" + str(frame + 1) + ".png"
            else:
                data_path = "BQMall_100/frame_" + str(frame) + ".png"
                target_path = "BQMall_100/frame_" + str(frame + 1) + ".png"

            createFrame(data_path,target_path,model,device,frame)
        
    
    else:
        # train set
        '''
        train_dataset = MyblockDataset(txt = 'data/train.txt', my_size = 16)
        train_loader = DataLoader(train_dataset, **train_kwargs)
        '''

        model = RFNet(1,1).to(device)

        optimizer = optim.Adadelta(model.parameters(), lr=args.lr) 
        epochs = args.epochs + 1
        scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)
        BestPsnr = 0  # 记录全部循环之后最好的psnr和ssim
        BestSsim = 0

        # 是否使用交叉验证
        if args.cross_flag == True:
            #####  交叉验证k取10 ######
            dataset = []
            txt = 'data/dataset.txt'
            fh = open(txt, 'r')
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                dataset.append(line)
            length = len(dataset) // args.k
            
            for fold in range(args.k):
                train_path = []
                val_path = []
                train_path, val_path = generateDataset(length, fold, dataset, train_path, val_path)
                train_dataset = MyblockDataset(path_list = train_path, my_size = 16)
                train_loader = DataLoader(train_dataset, **train_kwargs)
                test_dataset = MyDataset(path_list = val_path)  
                test_loader = DataLoader(test_dataset,**test_kwargs)

                for epoch in range(1, epochs): 
                    train(args, model, device, train_loader, optimizer, epoch, writer, fold)
                    scheduler.step()
                    temp_psnr,temp_ssim = test(args, model, device, test_loader, epoch, writer, fold)
                    if BestPsnr < temp_psnr:
                        BestPsnr = temp_psnr
                    if BestSsim < temp_ssim:
                        BestSsim = temp_ssim

        else :

            train_dataset = MyblockDataset(input_file = args.train_input_file, label_file = args.train_label_file)
            train_loader = DataLoader(train_dataset, **train_kwargs)
            test_dataset = MyDataset(test_file = args.test_file)  
            test_loader = DataLoader(test_dataset,**test_kwargs)
            for epoch in range(1, epochs): 
                train(args, model, device, train_loader, optimizer, epoch, writer)
                scheduler.step()
                temp_psnr, temp_ssim, temp_epoch = test(args, model, device, test_loader, epoch, writer)
                if BestPsnr < temp_psnr:
                    BestPsnr = temp_psnr
                    BestPsnrEpoch = temp_epoch
                if BestSsim < temp_ssim:
                    BestSsim = temp_ssim
                    BestSsimEpoch = temp_epoch
                
    
        print('\n after {} epoch,the best epoch {} ,the best psnr: {:.4f}'.format(args.epochs, BestPsnrEpoch, BestPsnr))
        print('\n after {} epoch,the best epoch {} ,the best ssim: {:.4f}'.format(args.epochs, BestSsimEpoch, BestSsim))
        # save model,optimizer,epoch
        args.save_model = True
        if args.save_model:
            state = {'net':model, 'optimizer':optimizer, 'epoch':epoch}
            torch.save(state, "model/model.pth")
 
        writer.close()

if __name__ == '__main__':
    main()

