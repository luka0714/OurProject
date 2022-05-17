import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import random
from PIL import Image
import cv2 as cv
from numpy import array,fromstring,frombuffer
import math
import random
import os

'''
    将训练数据对写入txt文件
'''
train_data_path = 'train_data/'
test_data_path = 'test_data/'
train_dir_list = []
test_dir_list = []
train_list_filename = []
test_list_filename = []

gtrain = os.walk(train_data_path)  
for path,dir_list,file_list in gtrain:  
    for dir_name in dir_list:  
        train_dir_list.append(os.path.join(path, dir_name))
        # print(os.path.join(path, dir_name) )

gtest = os.walk(test_data_path)  
for path,dir_list,file_list in gtest:  
    for dir_name in dir_list:  
        test_dir_list.append(os.path.join(path, dir_name))


#  train txtfile
train_txtfile = open('train_data/train_oneRef.txt',mode='w')

for file in train_dir_list:
    image_path_list = []
    write_list = []
    count = 1 
    # 遍历不同训练序列的文件夹
    for path, dir_list, file_list in os.walk(file):
        path = path + '/frame_'
        for file_name in file_list:
            
            # fpath = os.path.join(path, file_name)
            # image_path_list.append(fpath)

            if(count < 9):
                write_list.append(path + '00' + str(count) + '.png ' + path + '00' + str(count+1) + '.png\n')
            elif(count == 9):
                write_list.append(path + '00' + str(count) + '.png ' + path + '0' + str(count+1) + '.png\n')
            elif(count < 99):
                write_list.append(path + '0' + str(count) + '.png ' + path + '0' + str(count+1) + '.png\n')
            elif(count == 99):
                write_list.append(path + '0' + str(count) + '.png ' + path + str(count+1) + '.png\n')
            elif(count < len(file_list)):
                write_list.append(path + str(count) + '.png ' + path + str(count+1) + '.png\n')

            count += 1
    
    random.shuffle(write_list)
    for i in range(len(write_list)):
        train_txtfile.write(write_list[i])
    

# test txtfile
test_txtfile = open('train_data/test_oneRef.txt',mode='w')

for file in test_dir_list:
    image_path_list = []
    write_list = []
    count = 1 
    # 遍历不同训练序列的文件夹
    for path, dir_list, file_list in os.walk(file):
        path = path + '/frame_'
        for file_name in file_list:
            
            # fpath = os.path.join(path, file_name)
            # image_path_list.append(fpath)

            if(count < 9):
                write_list.append(path + '00' + str(count) + '.png ' + path + '00' + str(count+1) + '.png\n')
            elif(count == 9):
                write_list.append(path + '00' + str(count) + '.png ' + path + '0' + str(count+1) + '.png\n')
            elif(count < 99):
                write_list.append(path + '0' + str(count) + '.png ' + path + '0' + str(count+1) + '.png\n')
            elif(count == 99):
                write_list.append(path + '0' + str(count) + '.png ' + path + str(count+1) + '.png\n')
            elif(count < len(file_list)):
                write_list.append(path + str(count) + '.png ' + path + str(count+1) + '.png\n')

            count += 1
    
    random.shuffle(write_list)
    for i in range(len(write_list)):
        test_txtfile.write(write_list[i])
        

        for i in range (1, len(image_path_list)-1):
            write_list.append()



def i2a(mg):
    H,V = mg.size
    d = frombuffer(mg.tobytes(),dtype=np.uint8)
    d = d.reshape((V,H))
    return d

def CutToBlock(image1_path, target_path, image_block_list1, target_block_list, my_size):
    image1 = Image.fromarray(cv.imread(image1_path))
    target = Image.fromarray(cv.imread(target_path))
    image_gray1 = i2a(image1.convert('L'))
    target_gray = i2a(target.convert('L'))

    # image和target大小相同，算一遍height，width就行了
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
            
    return u,v    # u,v都是（block_height，block_width）大小的


