import os
from PIL import Image
from ffmpy3 import FFmpeg
 
inputPath = 'bqmall_prediction'
outputYUVPath = 'bqmall_prediction_yuv'
 
piclist = os.listdir(inputPath)
for pic in piclist:
    picpath = os.path.join(inputPath,pic)
    img = Image.open(picpath)
    in_wid,in_hei = img.size
    out_wid = in_wid//2*2
    out_hei = in_hei//2*2
    size = '{}x{}'.format(out_wid,out_hei)  #输出文件会缩放成这个大小
    purename = os.path.splitext(pic)[0]
    print(purename)
    outname = outputYUVPath + '/' + purename + '_' + size+ '.yuv'
    
    ff = FFmpeg(inputs={picpath:None},
                outputs={outname:'-s {} -pix_fmt yuv420p'.format(size)})
    print(ff.cmd)
    ff.run()
