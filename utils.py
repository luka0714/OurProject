import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image
import argparse
import math
from torchvision import transforms
import cv2 as cv
from numpy import array,fromstring


def i2a(mg):
    H,V = mg.size
    d = fromstring(mg.tobytes(),dtype=np.uint8)
    d = d.reshape((V,H))
    return d

def cal_psnr(im1, im2):
    mse = (np.abs(im1 - im2) ** 2).mean()
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr

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

# img是array类型
def addBoard(img):
    height = img.shape[0]
    width = img.shape[1]

    image = np.zeros((height+2,width+2))

    nheight = image.shape[0]  # 242
    nwidth = image.shape[1]  # 418
    # 填充中间部分
    for w in range(width):
        for h in range(height):
            image[1+h,1+w] = img[h,w]
    # 先padding四条边，分别是两个1*width和两个height*1
    toph = 0
    bottomh = nheight - 1
    leftw = 0
    leftr = nwidth - 1
    for tab in range(1,width + 1):
        image[toph, tab] = img[0, tab-1]
        image[bottomh, tab] = img[img.shape[0]-1, tab-1]
    for lar in range(1, height + 1):
        image[lar, leftw] = img[lar-1, 0]
        image[lar, leftr] = img[lar-1, img.shape[1]-1]
    # 再padding四个角
    image[0, 0] = image[1, 1]
    image[0, nwidth-1] = image[1, nwidth-2]
    image[nheight-1, 0] = image[nheight-2, 1]
    image[nheight-1, nwidth-1] = image[nheight-2, nwidth-2]

    return image


def CutToPadBlock(image, target, image_block_list, target_block_list, my_size, padding = 1):

    height = image.shape[0]  # 240
    width = image.shape[1]  # 416

    block_height = int(height/my_size) # 240/16 = 15
    block_width = int(width/my_size) # 416/16 = 26

    # x_mv是x分量上的，y_mv是y分量上的
    x_mv,y_mv = cal_grayimg_MV(target,image,my_size,height,width)
    
    imageB = addBoard(image)
   

    for i in range(0, block_height):
        for j in range(0, block_width):
            
            A = np.zeros((my_size + 2*padding, my_size + 2*padding)) # A是append到image_block里的  (18,18)
            # A = np.zeros((my_size, my_size))
            B = np.zeros((my_size, my_size)) # B是append到target_block里的 (16,16)
            # 先找到当前patch中最左上角的像素点坐标
            org_x = i*my_size
            org_y = j*my_size  # (org_x,org_y)是image中的最左上角的像素点坐标

            # 再通过(org_x,org_y)这个像素点和mv找到分像素点的坐标
            x_mv_f = org_x + y_mv[i,j]
            y_mv_f = org_y + x_mv[i,j]

            ref_x = math.floor(x_mv_f)
            ref_y = math.floor(y_mv_f)

            # 得到padding后的最左上角的像素点坐标
            ref_pad_x = ref_x - padding
            ref_pad_y = ref_y - padding

            # 将原坐标变换成在原图加上一层边界后的坐标，边界加了一层
            ref_pad_x = ref_pad_x + 1
            ref_pad_y = ref_pad_y + 1
            
            # imageB.shape[1]=418   imageB.shape[0]=242
            if (ref_pad_x >= 0 and ref_pad_y >= 0 and (ref_pad_x + my_size + 2*padding) <= (imageB.shape[0] - 1) and (ref_pad_y + my_size + 2*padding) <= (imageB.shape[1] - 1)):
                # 先填充A的中心部分
                for i_ in range(my_size):  # i_表示纵坐标，j_表示横坐标
                    for j_ in range(my_size):
                        A[padding + i_, padding + j_] = imageB[ref_pad_x + padding + i_, ref_pad_y + padding + j_]
                        B[i_, j_] = target[org_x + i_, org_y + j_] 
                        # A[i_, j_] = imageB[ref_pad_x + i_, ref_pad_y + j_]
                        # B[i_, j_] = target[org_x + i_, org_y + j_] 
                # 再填充A的padding部分
                for toph in range(padding):
                    for topw in range(my_size + 2*padding):
                        A[toph, topw] = imageB[ref_pad_x + toph, ref_pad_y + topw]
                for bottomh in range(padding):
                    for bottomw in range(my_size + 2*padding):
                        A[padding + my_size + bottomh, bottomw] = imageB[ref_pad_x + padding + my_size + bottomh, ref_pad_y + bottomw]
                for lefth in range(my_size):
                    for leftw in range(padding):
                        A[padding + lefth, leftw] = imageB[ref_pad_x + padding + lefth, ref_pad_y + leftw]
                for righth in range(my_size):
                    for rightw in range(padding):
                        A[padding + righth, padding + my_size + rightw] = imageB[ref_pad_x + padding + righth, ref_pad_y + padding + my_size + rightw]
                
                image_block_list.append(A)
                target_block_list.append(B)     
            
    return image_block_list, target_block_list

def cal_grayimg_MV(pic1_gray,pic2_gray,my_size,n,m):
    
    new_n = math.ceil(n//my_size)   # block_height
    new_m = math.ceil(m//my_size)   # block_weight

    np.set_printoptions(threshold=np.inf)

    m_th=pic2_gray[:,m-1]
    m_th = m_th.reshape(m_th.shape[0],1)
    n_th=pic2_gray[n-1,:]
    n_th = n_th.reshape(1,n_th.shape[0])
    mer1 = np.append(pic2_gray,m_th,axis=1)
    mer2 = np.append(n_th,0)
    mer2 = mer2.reshape(1,mer2.shape[0])

    pic2_ex = np.append(mer1,mer2,axis=0)
    #为了计算Ix、Iy在边缘进行周期延拓，原因可参考原理公式

    u = np.zeros((new_n,new_m))
    v = np.zeros((new_n,new_m))  #生成两个的零矩阵，每个区域对应一个(u,v)
    

    for fi in range(1,new_n+1):
        for fj in range(1,new_m+1):
            i = my_size*(fi-1)
            j = my_size*(fj-1) # i，j为每个块的最左上角像素位置

            A = np.zeros((2,2))    # LK方程等式右边的第一个矩阵
            b = np.zeros((2,1))    # LK方程等式右边的第二个矩阵
            for i_ in range(1,my_size):
                for j_ in range(1,my_size):
                # 这里两个循环是每个块内计算A、b矩阵进而求运动矢量    
                    fx = pic2_ex[(i+i_-1)+1,(j+j_-1)] - pic2_ex[(i+i_-1),(j+j_-1)]
                    fy = pic2_ex[(i+i_-1),(j+j_-1)+1] - pic2_ex[(i+i_-1),(j+j_-1)]
                    ft = pic2_ex[(i+i_-1),(j+j_-1)] - pic1_gray[(i+i_-1),(j+j_-1)]
                    
                    A[0,0] = A[0,0] + fx * fx
                    A[0,1] = A[0,1] + fx * fy
                    A[1,0] = A[0,1]
                    A[1,1] = A[1,1] + fy * fy
                    b[0,0] = b[0,0] - fx * ft
                    b[1,0] = b[1,0] - fy * ft

            if np.linalg.det(A) == 0:
                re = np.dot(np.linalg.pinv(A),b)
            else:
                re = np.dot(np.linalg.inv(A),b)

            u[fi-1,fj-1] = re[0,0]   
            v[fi-1,fj-1] = re[1,0]
            
    return u,v    # u,v都是（block_height，block_width）大小的

def CutToBlock(image1_path, target_path, image_block_org, target_block_org, image_block_list1, target_block_list, my_size):
    image1 = Image.fromarray(cv.imread(image1_path))
    target = Image.fromarray(cv.imread(target_path))
    image_gray1 = i2a(image1.convert('L'))
    target_gray = i2a(target.convert('L'))

    height = image_gray1.shape[0]  # 240
    width = image_gray1.shape[1]  # 416

    block_height = int(height/my_size) # 240/16 = 15
    block_width = int(width/my_size) # 416/16 = 26

    x_mv1,y_mv1 = cal_grayimg_MV(target_gray,image_gray1,my_size,height,width)  # (15, 26)

    A1 = np.zeros((my_size,my_size))  # 16*16
    B1 = np.zeros((my_size,my_size))
    A = np.zeros((my_size+2,my_size+2))  # padding=1
    B = np.zeros((my_size,my_size))
    
    for i in range(0,block_height):
        for j in range(0,block_width):

            x_mv_c1 = i + x_mv1[i,j]
            y_mv_c1 = j + y_mv1[i,j]  # x_mv_c1,y_mv_c1分别对应第一张参考图片的current block的第[i,j]块反向mv之后的x，y的坐标
            image_i1 = math.floor(x_mv_c1) 
            image_j1 = math.floor(y_mv_c1) # 得到第一张参考图片reference block的整数坐标位置
            
            if((image_i1 >= 0) and (image_i1 < block_height) and (image_j1 >= 0) and (image_j1 < block_width)):
                if((image_i1*my_size-1 >= 0) and (image_i1*my_size+my_size < height) and (image_j1*my_size-1 >= 0) and (image_j1*my_size+my_size < width)):

                    for i_ in range(my_size):
                        for j_ in range(my_size):
                            A1[i_,j_] = image_gray1[image_i1*my_size+i_, image_j1*my_size+j_]
                            B1[i_,j_] = target_gray[i*my_size+i_, j*my_size+j_]
                    image_block_org.append(A1)
                    target_block_org.append(B1)  

                
                    # 中间部分
                    for i_ in range(my_size):
                        for j_ in range(my_size):
                            # A1[i_,j_] = image_gray1[image_i1*my_size+i_, image_j1*my_size+j_]
                            A[i_+1, j_+1] = image_gray1[image_i1*my_size+i_, image_j1*my_size+j_]
                            B[i_,j_] = target_gray[i*my_size+i_, j*my_size+j_]

                    # 四条长度为my_size的边
                    for tab in range(my_size):
                        A[0, tab+1] = image_gray1[image_i1*my_size-1,image_j1*my_size+tab]
                        A[my_size+1, tab+1] = image_gray1[image_i1*my_size+my_size,image_j1*my_size+tab]
                    for lar in range(my_size):
                        A[lar+1, 0] = image_gray1[image_i1*my_size+lar, image_j1*my_size-1]
                        A[lar+1, my_size+1] = image_gray1[image_i1*my_size+lar, image_j1*my_size+my_size]
                    # 四个角
                    A[0, 0] = image_gray1[image_i1*my_size-1,image_j1*my_size-1]
                    A[0, my_size+1] = image_gray1[image_i1*my_size-1, image_j1*my_size+my_size]
                    A[my_size+1, 0] = image_gray1[image_i1*my_size+my_size, image_j1*my_size-1]
                    A[my_size+1, my_size+1] = image_gray1[image_i1*my_size+my_size, image_j1*my_size+my_size]
                    
                    image_block_list1.append(A)
                    target_block_list.append(B)          

    return image_block_org, target_block_org, image_block_list1, target_block_list


def CutToBlockorg(image1_path, target_path, image_block_list1, target_block_list, my_size):
    image1 = Image.fromarray(cv.imread(image1_path))
    target = Image.fromarray(cv.imread(target_path))
    image_gray1 = i2a(image1.convert('L'))
    target_gray = i2a(target.convert('L'))

    # image和target大小相同，算一遍height，width就行了
    height = image_gray1.shape[0]  # 
    width = image_gray1.shape[1]   # 
    block_height = int(height/my_size) # 240/16 = 15
    block_width = int(width/my_size) # 416/16 = 26

    x_mv1,y_mv1 = cal_grayimg_MV(target_gray,image_gray1,my_size,height,width)
    
    Ao = np.zeros((my_size,my_size))
    Bo = np.zeros((my_size,my_size))
    for i in range(0,block_height):
        for j in range(0,block_width):     

            
            x_mv_c1 = i + x_mv1[i,j]
            y_mv_c1 = j + y_mv1[i,j]  # x_mv_c1,y_mv_c1分别对应第一张参考图片的current block的第[i,j]块反向mv之后的x，y的坐标
            image_i1 = math.floor(x_mv_c1) 
            image_j1 = math.floor(y_mv_c1) # 得到第一张参考图片reference block的整数坐标位置
            
            if((image_i1 >= 0) and (image_i1 < block_height) and (image_j1 >= 0) and (image_j1 < block_width)):
                if((image_i1*my_size-1 >= 0) and (image_i1*my_size+my_size < height) and (image_j1*my_size-1 >= 0) and (image_j1*my_size+my_size < width)):
                    for i_ in range(my_size):
                        for j_ in range(my_size):
                            Ao[i_,j_] = image_gray1[image_i1*my_size+i_, image_j1*my_size+j_]
                            Bo[i_,j_] = target_gray[i*my_size+i_, j*my_size+j_]

                    image_block_list1.append(Ao)
                    target_block_list.append(Bo)          

    return image_block_list1, target_block_list

def createFrame(input_path, target_path, model, device, frame):
    model.eval()   
    loss = 0
    psnr = 0
    ssim = 0
    with torch.no_grad(): 
        data = np.array(Image.open(input_path).convert('L')).astype(np.float32)
        target = np.array(Image.open(target_path).convert('L')).astype(np.float32)
        data = transforms.ToTensor()(data)
        target = transforms.ToTensor()(target)
        data = torch.unsqueeze(data, dim=0)  
        target = torch.unsqueeze(target, dim=0)
        
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.mse_loss(output, target).item()
        input_image = data.cpu().numpy().reshape(data.shape[2],data.shape[3])
        output_2d = output.cpu().numpy().reshape(output.shape[2],output.shape[3])
        target_2d = target.cpu().numpy().reshape(target.shape[2],target.shape[3])
        print(input_image)
        print(output_2d)
        print(target_2d)

    psnr = cal_psnr(output_2d, target_2d)
    ssim = cal_ssim(output_2d, target_2d)
    
    print("\n psnr:{:.4f}".format(psnr))
    print("\n ssim:{:.4f}".format(ssim))

    im = Image.fromarray(output_2d.astype(np.uint8))

    # VRCNN
    if frame < 10:
        save_path = "bqmall_prediction_VRCNN/poc00" + str(frame) + "s.png"
    elif frame < 100:
        save_path = "bqmall_prediction_VRCNN/poc0" + str(frame) + "s.png"
    else:
        save_path = "bqmall_prediction_VRCNN/poc" + str(frame) + "s.png"
    
    im.save(save_path)

