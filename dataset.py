import numpy as np
import cv2 as cv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os 
import math
from numpy import array,fromstring
import random
from utils import addBoard

def i2a(mg):
    H,V = mg.size
    d = fromstring(mg.tobytes(),dtype=np.uint8)
    d = d.reshape((V,H))
    return d

def CutToBlock(image1_path, target_path, image_block_list1, target_block_list, my_size):
    image1 = Image.fromarray(cv.imread(image1_path))
    target = Image.fromarray(cv.imread(target_path))
    image_gray1 = i2a(image1.convert('L'))
    target_gray = i2a(target.convert('L'))

    height = image_gray1.shape[0]
    width = image_gray1.shape[1]
    block_height = int(height/my_size) # 240/16 = 15
    block_width = int(width/my_size) # 416/16 = 26

    x_mv1,y_mv1 = cal_grayimg_MV(target_gray,image_gray1,my_size,height,width)
    
    A1 = np.zeros((my_size,my_size)) #16*16
    B = np.zeros((my_size,my_size))

    for i in range(0,block_height):
        for j in range(0,block_width):

            x_mv_c1 = i + x_mv1[i,j]
            y_mv_c1 = j + y_mv1[i,j]  # x_mv_c1,y_mv_c1分别对应第一张参考图片的current block的第[i,j]块反向mv之后的x，y的坐标
            image_i1 = math.floor(x_mv_c1) 
            image_j1 = math.floor(y_mv_c1) # 得到第一张参考图片reference block的整数坐标位置
            
            if((image_i1 >= 0) and (image_i1 < block_height) and (image_j1 >= 0) and (image_j1 < block_width)):
                for i_ in range(my_size):
                    for j_ in range(my_size):
                        A1[i_,j_] = image_gray1[image_i1*my_size+i_, image_j1*my_size+j_]
                        B[i_,j_] = target_gray[i*my_size+i_, j*my_size+j_]
                image_block_list1.append(A1)
                target_block_list.append(B)          

    return image_block_list1, target_block_list

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
            
    return u,v    # u,v都是（block_height，block_width）大小的  (15,26)

class MyblockDataset (Dataset):   
    # 构造函数带有默认参数
    def __init__(self, input_file='', label_file='', path_list=[], transform=transforms.ToTensor(), target_transform=transforms.ToTensor()):
        
        image_block_list1 = []
        target_block_list = []
        if input_file != '' and label_file != '': 
            image_block_list1 = np.load(input_file)
            target_block_list = np.load(label_file)
        else:
            for line in range(len(path_list)-1):
                words = path_list[line].split()
                image_block_list1,target_block_list = CutToBlock(words[0],words[1],image_block_list1,target_block_list,my_size)
        

        # print(len(image_block_list1))
        self.imgs1 = image_block_list1
        self.target = target_block_list
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        
        image_block1 = self.imgs1[index]
        target_block = self.target[index]     

        # image_block和target_block都是numpy.ndarray类型的
        sample_block = {'image1': image_block1, 'target': target_block}  

        # 范围是[0,255]
        if self.transform:
            sample_block['image1'] = self.transform(sample_block['image1'])
        if self.target_transform:
            sample_block['target'] = self.target_transform(sample_block['target'])

        
        return sample_block

    def __len__(self):
        return len(self.imgs1)

class MyDataset (Dataset):   
    # 构造函数带有默认参数
    def __init__(self, test_file='', path_list=[], transform=transforms.ToTensor(), target_transform=transforms.ToTensor()):
        
        imgs_list1 = []
        target_list = []
        if test_file != '':
            fh = open(test_file,'r')
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs_list1.append(words[0])
                target_list.append(words[1])  
        else:
            for line in range(len(path_list)-1):
                words = path_list[line].split()
                imgs_list1.append(words[0])
                target_list.append(words[1])  

        self.imgs1 = imgs_list1
        self.target = target_list
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path1 = self.imgs1[index]
        target_path = self.target[index]
        image1 = Image.fromarray(cv.imread(img_path1))
        target = Image.fromarray(cv.imread(target_path))
        image_gray1 = np.array(image1.convert('L')).astype(np.float32)
        # pad_image = addBoard(image_gray1).astype(np.float32)
        target_gray = np.array(target.convert('L')).astype(np.float32)

        sample = {'image1': image_gray1, 'target': target_gray} 

        # 范围[0,255]
        if self.transform:
            sample['image1'] = self.transform(sample['image1'])
        if self.target_transform:
            sample['target'] = self.target_transform(sample['target'])

        return sample

    def __len__(self):
        return len(self.imgs1)


def Cut(preimg_path, curimg_path, preblock, curblock, my_size):
    preimg = Image.open(preimg_path).convert('L')
    curimg = Image.open(curimg_path).convert('L')
    size = preimg.size 
    preimg = np.array(preimg)
    curimg = np.array(curimg)
    x_mv,y_mv = cal_grayimg_MV(curimg,preimg,my_size,size[1],size[0])   

    A = np.zeros((my_size,my_size)) #16*16
    B = np.zeros((my_size,my_size))
    block_width = size[0]//my_size
    block_height = size[1]//my_size

    for i in range(0,block_height):
        for j in range(0,block_width):
            x_mv_c = i + x_mv[i,j]
            y_mv_c = j + y_mv[i,j]  # x_mv_c1,y_mv_c1分别对应第一张参考图片的current block的第[i,j]块反向mv之后的x，y的坐标
            image_i = math.floor(x_mv_c) 
            image_j = math.ceil(y_mv_c) # 得到第一张参考图片reference block的整数坐标位置
            
            if((image_i >= 0) and (image_i < block_height) and (image_j >= 0) and (image_j < block_width)):
                for i_ in range(my_size):
                    for j_ in range(my_size):
                        A[i_,j_] = preimg[image_i*my_size+i_, image_j*my_size+j_]
                        B[i_,j_] = curimg[i*my_size+i_, j*my_size+j_]
                preblock.append(A)
                curblock.append(B)          
                #psnr_block = batch_PSNR(A, B)
                #print(psnr_block)
    return preblock, curblock

class MyTrainDataset(Dataset):
    def __init__(self, root):
        '''
            初始化
        '''
        preblock = []  # preblock为input
        curblock = []  # curblock为target
        size = 16 # block设置为16*16
        self.transforms = transforms.ToTensor()
        
        imgs_path = [os.path.join(root, img) for img in os.listdir(root)]
        for img in range(len(imgs_path)-1):  # 取值从 0 ~ len(self.imgs_path)-1-1 
            preblock, curblock = Cut(imgs_path[img], imgs_path[img+1],preblock, curblock, size)

        self.preblock = preblock
        self.curblock = curblock

    def __getitem__(self,index):
        '''
            取数据 
        '''
        pre_block = self.preblock[index].astype(np.uint8)
        cur_block = self.curblock[index].astype(np.uint8)
        sample = {'pre_block':pre_block, 'cur_block':cur_block}       
        sample['pre_block'] = self.transforms(sample['pre_block'])
        sample['cur_block'] = self.transforms(sample['cur_block'])
        return sample

    def __len__(self):
        return len(self.preblock)

# val为image的数据集
class MyValandTestDataset(Dataset):
    def __init__(self, root):
        '''
            初始化
        '''
        preimage_path = []
        curimage_path = []
        self.transforms = transforms.ToTensor()
        imgs_path = [os.path.join(root, img) for img in os.listdir(root)]
        for i in range(len(imgs_path)-1):
            preimage_path.append(imgs_path[i])
            curimage_path.append(imgs_path[i+1])

        self.preimage_path = preimage_path
        self.curimage_path = curimage_path
        self.imgs_path = imgs_path

    def __getitem__(self, index):
        '''
            取数据
        '''
        gpreimage = Image.open(self.preimage_path[index]).convert('L')
        gcurimage = Image.open(self.curimage_path[index]).convert('L')
        gpreimage = np.array(gpreimage).astype(np.uint8)
        gcurimage = np.array(gcurimage).astype(np.uint8)
        sample = {'pre_image':gpreimage, 'cur_image':gcurimage}
        sample['pre_image'] = self.transforms(sample['pre_image'])
        sample['cur_image'] = self.transforms(sample['cur_image'])
        return sample

    def __len__(self):
        return len(self.preimage_path)

# val为block的数据集
class MyValandTestBlockDataset(Dataset):
    def __init__(self, root):
        '''
            初始化
        '''
        preblock = []  # preblock为input
        curblock = []  # curblock为target
        size = 16 # block设置为16*16
        self.transforms = transforms.ToTensor()
        
        imgs_path = [os.path.join(root, img) for img in os.listdir(root)]
        for img in range(len(imgs_path)-1):  # 取值从 0 ~ len(self.imgs_path)-1-1 
            preblock, curblock = Cut(imgs_path[img], imgs_path[img+1],preblock, curblock, size)
        self.preblock = preblock
        self.curblock = curblock

    def __getitem__(self,index):
        '''
            取数据 
        '''
        pre_block = self.preblock[index].astype(np.uint8)
        cur_block = self.curblock[index].astype(np.uint8)
        sample = {'pre_val_block':pre_block, 'cur_val_block':cur_block}       
        sample['pre_val_block'] = self.transforms(sample['pre_val_block'])
        sample['cur_val_block'] = self.transforms(sample['cur_val_block'])
        return sample

    def __len__(self):
        return len(self.preblock)

# 乱序排放
class MyDDataset(Dataset):
    def __init__(self, root, mark):
        '''
            初始化取数据
        '''
        size = 32 # block设置为16*16
        self.transforms = transforms.ToTensor()
        self.mark = mark
        imgs_path = [os.path.join(root, img) for img in os.listdir(root)]
        pre_process = []
        for i in range(len(imgs_path)-1):
            pre_process.append(imgs_path[i] + "++" + imgs_path[i+1])
        random.shuffle(pre_process)

        if mark == "train":
            preblock = []  # preblock为input
            curblock = []  # curblock为target
            for img in range(int(len(pre_process)*0.95)):  # 取值从 0 ~ len(self.imgs_path)-1-1 
                prepro_imgpath = pre_process[img].split('++')
                preblock, curblock = Cut(prepro_imgpath[0], prepro_imgpath[1], preblock, curblock, size)
            self.preblock = preblock
            self.curblock = curblock
            self.length = len(self.preblock)
        if mark == "val":
            preimage_path = []
            curimage_path = []
            for val in range(int(len(pre_process)*0.95), int(len(pre_process))):
                prepro_imgpath = pre_process[val].split('++')
                preimage_path.append(prepro_imgpath[0])
                curimage_path.append(prepro_imgpath[1])
            self.preimage_path = preimage_path
            self.curimage_path = curimage_path
            self.length = len(self.preimage_path)

    def __getitem__(self,index):
        '''
            将数据存成字典的形式
        '''
        if self.mark == "train":
            pre_block = self.preblock[index].astype(np.uint8)
            cur_block = self.curblock[index].astype(np.uint8)
            sample = {'pre_block':pre_block, 'cur_block':cur_block}       
            sample['pre_block'] = self.transforms(sample['pre_block'])
            sample['cur_block'] = self.transforms(sample['cur_block'])
        if self.mark == "val":
            gpreimage = Image.open(self.preimage_path[index]).convert('L')
            gcurimage = Image.open(self.curimage_path[index]).convert('L')
            gpreimage = np.array(gpreimage).astype(np.uint8)
            gcurimage = np.array(gcurimage).astype(np.uint8)
            sample = {'pre_image':gpreimage, 'cur_image':gcurimage}
            sample['pre_image'] = self.transforms(sample['pre_image'])
            sample['cur_image'] = self.transforms(sample['cur_image'])
        return sample

    def __len__(self):
        return self.length