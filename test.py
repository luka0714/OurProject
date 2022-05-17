import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import math
from PIL import Image
from torchvision import transforms
import os

def cal_psnr(im1, im2):
    
    mse = (np.abs(im1 - im2) ** 2).mean()
    psnr = 10 * np.log10(255 * 255 / mse)
    
    return psnr

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255**2/mse)

def cal_ssim(im1,im2):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim

def test(args, model, device, test_loader, epoch, writer, fold = 0):
    model.eval()   
    item_loss = 0
    psnr = 0
    ssim = 0
    every_psnr = 0
    every_ssim = 0
    best_psnr = 0
    best_ssim = 0
    worse_psnr = 100
    with torch.no_grad(): 
        for test_idx,test_data in enumerate(test_loader):
            data1 = test_data['image1'] 
            target = test_data['target'] 
            data1, target = data1.to(device), target.to(device)
            
            output = model(data1)

            test_loss = F.mse_loss(output,target)
            item_loss += test_loss.item()

            input_image = data1.cpu().numpy().reshape(data1.shape[2],data1.shape[3])
            output_2d = output.cpu().numpy().reshape(output.shape[2],output.shape[3])
            target_2d = target.cpu().numpy().reshape(target.shape[2],target.shape[3])  
            
            every_psnr = cal_psnr(output_2d,target_2d)
            every_ssim = cal_ssim(output_2d,target_2d)
            if(every_psnr > best_psnr):  # 取最好的psnr
                best_psnr = every_psnr
                best_epoch = epoch
                # 记录特征图
                if args.flag == True:
                    conv1 = model.conv1_feature
                    conv2 = model.conv2_feature
                    RFB1 = model.RFB1_feature
                    RFB2 = model.RFB2_feature
                    RFB3 = model.RFB3_feature
                # 记录输出图像
                inp_image = Image.fromarray(input_image.astype(np.uint8))
                out_image = Image.fromarray(output_2d.astype(np.uint8))
                itarget = Image.fromarray(target_2d.astype(np.uint8))
            if(every_ssim > best_ssim):  # 取最好的ssim
                best_ssim = every_ssim
            if(every_psnr < worse_psnr): # 取最差的psnr
                worse_psnr = every_psnr
                imagew = Image.fromarray(output_2d)
                itargetw = Image.fromarray(target_2d)

            psnr += every_psnr
            ssim += every_ssim

    # 输出特征图
    if args.flag == True:
        if args.cross_flag == True:
            epochs = args.epochs*fold + epoch
        else:
            epochs = epoch
        if epochs % 10 == 0:
            epoch_path = args.outdir + '\\epoch' + str(epochs)
            if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)
            layer1_path = epoch_path + '\\layer1'
            layer2_path = epoch_path + '\\layer2'
            layer3_path = epoch_path + '\\layer3'
            layer4_path = epoch_path + '\\layer4'
            layer5_path = epoch_path + '\\layer5'
            if not os.path.exists(layer1_path):
                os.makedirs(layer1_path)
            if not os.path.exists(layer2_path):
                os.makedirs(layer2_path)
            if not os.path.exists(layer3_path):
                os.makedirs(layer3_path)
            if not os.path.exists(layer4_path):
                os.makedirs(layer4_path)
            if not os.path.exists(layer5_path):
                os.makedirs(layer5_path)
            creatFeatureMap(layer1_path, conv1, inp_image, out_image,itarget)
            creatFeatureMap(layer2_path, conv2, inp_image, out_image,itarget)
            creatFeatureMap(layer3_path, RFB1, inp_image, out_image,itarget)
            creatFeatureMap(layer4_path, RFB2, inp_image, out_image,itarget)
            creatFeatureMap(layer5_path, RFB3, inp_image, out_image,itarget)       
    
    psnr /= len(test_loader.dataset)
    ssim /= len(test_loader.dataset)
    item_loss /= len(test_loader.dataset)  # 损失的平均值

    print('\nTest set:  Average loss: {:.4f}'.format(item_loss))
    print('\nTest set:  worse psnr: {:.4f}'.format(worse_psnr))
    print('\nTest set:  best psnr: {:.4f}'.format(best_psnr))
    print('\nTest set:  Average psnr: {:.4f}'.format(psnr))
    print('\nTest set:  Average ssim: {:.4f}'.format(ssim))
    
    writer.add_scalar('Test/Loss', item_loss,args.epochs * fold + epoch)
    writer.add_scalar('best psnr', best_psnr,args.epochs * fold + epoch)
    writer.add_scalar('Test/psnr', psnr,args.epochs * fold + epoch)
    writer.add_scalar('Test/ssim', ssim,args.epochs * fold + epoch)

    return best_psnr, best_ssim, best_epoch

def creatFeatureMap(path, feature, input_image, out_image,itarget):
    input_image.save(path + '\\1input_image.png')
    out_image.save(path + '\\1output_image.png')
    itarget.save(path + '\\1target_image.png')
    layer_feature = torch.squeeze(feature, dim=0)
    for x in range(layer_feature.shape[0]):
        layer_features = layer_feature[x,:,:].cpu()  # shape:torch.Size([240, 416])
        layer_features = np.array(layer_features)
        layer_features = Image.fromarray(layer_features.astype(np.uint8))
        layer_features.save(path + '\\filter_f' + str(x+1) + '.png')