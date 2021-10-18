# Spatial-Angular Attention Network for Light Field Reconstruction
### [Project Page](http://www.liuyebin.com/localtrans/localtrans.html) | [Paper](https://arxiv.org/abs/2106.04067) | [Video](https://www.bilibili.com/video/BV1zq4y1W7Uq/)

[Gaochang Wu](https://gaochangwu.github.io/)<sup>1</sup>,
[Yingqian Wang](https://yingqianwang.github.io/)<sup>2</sup>,
[Yebin Liu](http://www.liuyebin.com/)<sup>3</sup>,
[Lu Fang](http://luvision.net/)<sup>4</sup>,
[Tianyou Chai](http://www.sapi.neu.edu.cn/)<sup>1</sup><br>

<sup>1</sup>State Key Laboratory of Synthetical Automation for Process Industries, Northeastern University <br> 
<sup>2</sup>College of Electronic Science and Technology, Nation University of Defense Technology (NUDT) <br> 
<sup>3</sup>Department of Automation, Tsinghua University <br>
<sup>4</sup>Tsinghua-Berkeley Shenzhen Institute <br>

## Note for Code
1. Environment -Python 3.7.4, tensorflow-gpu==1.13.1 <br>

2. You should first upload your light fields to "./Datasets/". <br>

3. The code for 3D light field (1D angular and 2D spatial) reconstruction is "main3d.py". Recommend using the model with upsampling scale \alpha=3 for x8 or x9 reconstruction, and the model with upsampling scale \alpha=4 for x16 reconstruction. <br>

4. Please cite our paper if it helps, thank you! <br>
