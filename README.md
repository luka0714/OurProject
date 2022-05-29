# DILATED CONVOLUTIONAL NEURAL NETWORK-BASED DEEP REFERENCE PICTURE GENERATION FOR VIDEO COMPRESSION
This is the implemention of the paper "DILATED CONVOLUTIONAL NEURAL NETWORK-BASED DEEP REFERENCE PICTURE GENERATION FOR VIDEO COMPRESSION".

### Abstract ###
Motion estimation and motion compensation are indispensable parts of inter prediction in video coding. Since the motion vector of objects is mostly in fractional pixel units, original reference pictures may not accurately provide a suitable reference for motion compensation. In this paper, we propose a deep reference picture generator which can create a picture that is more relevant to the current encoding frame, thereby further reducing temporal redundancy and improving video compression efficiency. Inspired by the recent progress of Convolutional Neural Network(CNN), this paper proposes to use a dilated CNN to build the generator. Moreover, we insert the generated deep picture into Versatile Video Coding(VVC) as a reference picture and perform a comprehensive set of experiments to evaluate the effectiveness of our network on the latest VVC Test Model–VTM. The experimental results demonstrate that our proposed method achieves on average 9.7% bit saving compared with VVC under low-delay P configuration.

Paper URL: https://arxiv.org/abs/2202.05514

<img src="/pic/img1.png" width="50%" height="50%">

### Citation ###

If you find the code and datasets useful in your research, please cite: 

<pre><code>@inproceedings{tian2022dilated,   
  title={Dilated convolutional neural network-based deep reference picture generation for video compression},
  author={Tian, Haoyue and Gao, Pan and Wei, Ran and Paul, Manoranjan},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={2824--2828},
  year={2022},
  organization={IEEE}
}</code></pre>


### Requirement ###

* python = 3.7.6 
* pytorch = 1.8.0 
* torchvision = 0.10.0 
* cudatoolkit = 10.2 
* cudnn = 7.6.5  
* NVIDIA GPU = 2080ti
* VTM = 10.0

### Train Model ###

We use BlowingBubbles in HEVC test sequence as training dataset. Download the dataset： \
链接：https://pan.baidu.com/s/1VYj_Iyh-hU9R8GLnCLfKNA    提取码：luka  \
In the data folder, we have extracted the BlowingBubbles video sequence into 500 frames of images with a resolution of 416*240. Of course you can try with other video sequences.

Run the main file to train the model.
<pre><code>python main.py
</code></pre>

### Embed in VVC ###
Our approach is to use the previous frame of the current frame as the network input, and the purpose is to output a picture that is more similar to the current frame through the trained network. Then, we add the model predicted picture into the original reference list. We used the encoder of VVC reference software VTM (version 10.0) for experiments, we follow the VVC common test conditions and use LDP configuration to do compression performance test, under 4 quantization parameters (QPs): 22, 27, 32, and 37.

In VTM, we modified the EncGOP.cpp file to add the depth reference image output by the network to the reference image list.

### Contact ###
<a href="tianhy@nuaa.edu.cn">Haoyue Tian</a>
<a href="pan.gao@nuaa.edu.cn">Pan Gao</a>

