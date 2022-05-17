
import cv2
import numpy as np

from PIL import Image

class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = int(self.width * self.height * 3 / 2)
        self.f = open(filename, 'rb')
        self.shape = (int(self.height*1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print(str(e))
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return ret, bgr

import cv2
import numpy as np
 
 
def yuv2bgr(filename, height, width, startfrm):
    """
    :param filename: 待处理 YUV 视频的名字
    :param height: YUV 视频中图像的高
    :param width: YUV 视频中图像的宽
    :param startfrm: 起始帧
    :return: None
    """
    fp = open(filename, 'rb')

    framesize = height * width * 3 // 2 # 一帧图像所含的像素个数
    h_h = height // 2
    h_w = width // 2

    fp.seek(0, 2) # 设置文件指针到文件流的尾部
    ps = fp.tell() # 当前文件指针位置
    numfrm = ps // framesize # 计算输出帧数
    fp.seek(framesize * startfrm, 0)

    for i in range(numfrm - startfrm):
        Yt = np.zeros(shape=(height, width), dtype='uint8', order='C')
        Ut = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')
        Vt = np.zeros(shape=(h_h, h_w), dtype='uint8', order='C')

        for m in range(height):
            for n in range(width):
                Yt[m, n] = ord(fp.read(1))
        for m in range(h_h):
            for n in range(h_w):
                Ut[m, n] = ord(fp.read(1))
        for m in range(h_h):
            for n in range(h_w):
                Vt[m, n] = ord(fp.read(1))

        img = np.concatenate((Yt.reshape(-1), Ut.reshape(-1), Vt.reshape(-1)))
        img = img.reshape((height * 3 // 2, width)).astype('uint8') # YUV 的存储格式为：NV12（YYYY UV）

        # 由于 opencv 不能直接读取 YUV 格式的文件, 所以要转换一下格式
        bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12) # 注意 YUV 的存储格式
        cv2.imwrite('BasketballPass/%d.jpg' % (i + 1), bgr_img)
        print("Extract frame %d " % (i + 1))

    fp.close()
    print("job done!")
    return None
 
# return a n * 3 matrix
def get_all_frames(filename, height, width, num_frames):
    y = bytes()
    u = bytes()
    v = bytes()
    with open(filename, "rb") as f:
        for i in range(num_frames):
            y = f.read(height * width)
            u = f.read(height * width)
            v = f.read(height * width)
            y = list(y)
            u = list(u)
            v = list(v)
            y = np.asarray(y)
            u = np.asarray(u)
            v = np.asarray(v)
            yield np.vstack([y, u, v]).T, i

def yuv_to_rgb(yuv_frame):
    rgb_frame = np.copy(yuv_frame)
    v1 = [1.000, 0.001, 1.574]
    v2 = [1.000, -0.187, -0.469]
    v3 = [1.000, 1.856, 0.001]    
    mat1 = np.array([v1, v2, v3])
    mat1 = mat1.T
    v4 = np.array([[16, 128, 128]], dtype="uint8")
    rgb_frame -= v4
    rgb_frame = np.clip(np.matmul(rgb_frame, mat1), 0, 255)
    return np.uint8(rgb_frame.reshape(yuv_frame.shape))

def save_rgb(filename, rgb, height, width):
    rgb_frame = rgb.reshape(height, width, 3)
    img = Image.fromarray(rgb_frame)
    img.save(filename)

# read frames from a yuv file and save the frame to image files
# yuv_filename: the yuv file to convert
# height, width: the height and the width of the yuv file
# num_frames: how many frames of the yuv file to convert
# format: such as "jpg", "bmp"
# img_filename: a string used in image files
def convertYUV2Img(yuv_filename, height, width, num_frames, format, img_filename):
    for yuv, i in get_all_frames(yuv_filename, height, width, num_frames):
        rgb = yuv_to_rgb(yuv)
        tmp_filename = img_filename + ("-%d.%s" % (i, format))
        save_rgb(tmp_filename, rgb, height, width)


  
from PIL import Image


def yuv420_to_rgb888(width, height, yuv):
    # function requires both width and height to be multiples of 4
    if (width % 4) or (height % 4):
        raise Exception("width and height must be multiples of 4")
    rgb_bytes = bytearray(width*height*3)

    red_index = 0
    green_index = 1
    blue_index = 2
    y_index = 0

    for row in range(0,height):
        u_index = width * height + (row//2)*(width//2)
        v_index = u_index + (width*height)//4

        for column in range(0,width):
            Y = yuv[y_index]
            U = yuv[u_index]
            V = yuv[v_index]
            C = (Y - 16) * 298
            D = U - 128
            E = V - 128
            R = (C + 409*E + 128) // 256
            G = (C - 100*D - 208*E + 128) // 256
            B = (C + 516 * D + 128) // 256

            R = 255 if (R > 255) else (0 if (R < 0) else R)
            G = 255 if (G > 255) else (0 if (G < 0) else G)
            B = 255 if (B > 255) else (0 if (B < 0) else B)

            rgb_bytes[red_index] = R
            rgb_bytes[green_index] = G
            rgb_bytes[blue_index] = B

            u_index += (column % 2)
            v_index += (column % 2)
            y_index += 1
            red_index += 3
            green_index += 3
            blue_index += 3

    return rgb_bytes




def testConversion(source, dest):
    print("opening file")
    f = open(source, "rb")
    yuv = f.read()
    f.close()

    print("read file")
    rgb_bytes = yuv420_to_rgb888(1920,1088, yuv)
    # cProfile.runctx('yuv420_to_rgb888(1920,1088, yuv)', {'yuv420_to_rgb888':yuv420_to_rgb888}, {'yuv':yuv})
    print("finished conversion. Creating image object")

    img = Image.frombytes("RGB", (1920,1088), bytes(rgb_bytes))
    print("Image object created. Starting to save")

    img.save(dest, "JPEG")
    img.close()
    print("Save completed")

import cv2
import numpy as np


def yuv2bgr(file_name, height, width, start_frame):
    """
    :param file_name: 待处理 YUV 视频的名字
    :param height: YUV 视频中图像的高
    :param width: YUV 视频中图像的宽
    :param start_frame: 起始帧
    :return: None
    """

    fp = open(file_name, 'rb')
    fp.seek(0, 2)  # 设置文件指针到文件流的尾部 + 偏移 0
    fp_end = fp.tell()  # 获取文件尾指针位置

    frame_size = height * width * 3 // 2  # 一帧图像所含的像素个数
    num_frame = fp_end // frame_size  # 计算 YUV 文件包含图像数
    print("This yuv file has {} frame imgs!".format(num_frame))
    fp.seek(frame_size * start_frame, 0)  # 设置文件指针到文件流的起始位置 + 偏移 frame_size * startframe
    print("Extract imgs start frame is {}!".format(start_frame + 1))

    for i in range(num_frame - start_frame):
        yyyy_uv = np.zeros(shape=frame_size, dtype='uint8', order='C')
        for j in range(frame_size):
            yyyy_uv[j] = ord(fp.read(1))  # 读取 YUV 数据，并转换为 unicode

        img = yyyy_uv.reshape((height * 3 // 2, width)).astype('uint8')  # NV12 的存储格式为：YYYY UV 分布在两个平面（其在内存中为 1 维）
        bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV420p2BGR )  # 由于 opencv 不能直接读取 YUV 格式的文件, 所以要转换一下格式，支持的转换格式可参考资料 5
        cv2.imwrite('BasketballPass/frame_{}.png'.format(i + 1), bgr_img)  # 改变后缀即可实现不同格式图片的保存(jpg/bmp/png...)
        print("Extract frame {}".format(i + 1))

    fp.close()
    print("job done!")
    return None


if __name__ == '__main__':
	yuv2bgr(file_name='sequence/ClassD_416x240/BasketballPass.yuv', height=240, width=416, start_frame=0)
