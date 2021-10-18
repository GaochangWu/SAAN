# Spatial-Angular Attention Network for Light Field Reconstruction
### [Project Page](https://gaochangwu.github.io/SAAN/SAA-Net.html) | [Paper](https://arxiv.org/pdf/2007.02252) | [Video](https://www.bilibili.com/video/BV1VK411H7Z1/)

[Gaochang Wu](https://gaochangwu.github.io/)<sup>1</sup>,
[Yingqian Wang](https://yingqianwang.github.io/)<sup>2</sup>,
[Yebin Liu](http://www.liuyebin.com/)<sup>3</sup>,
[Lu Fang](http://luvision.net/)<sup>4</sup>,
[Tianyou Chai](http://www.sapi.neu.edu.cn/)<sup>1</sup><br>

<sup>1</sup>State Key Laboratory of Synthetical Automation for Process Industries, Northeastern University <br> 
<sup>2</sup>College of Electronic Science and Technology, Nation University of Defense Technology (NUDT) <br> 
<sup>3</sup>Department of Automation, Tsinghua University <br>
<sup>4</sup>Tsinghua-Berkeley Shenzhen Institute <br>

## Abstract
![Teaser Image](https://gaochangwu.github.io/image/SAAN.jpg)

Typical learning-based light field reconstruction methods demand in constructing a large receptive field by deepening their networks to capture correspondences between input views. In this paper, we propose a spatial-angular attention network to perceive non-local correspondences in the light field, and reconstruct high angular resolution light field in an end-to-end manner. Motivated by the non-local attention mechanism, a spatial-angular attention module specifically for the high-dimensional light field data is introduced to compute the response of each query pixel from all the positions on the epipolar plane, and generate an attention map that captures correspondences along the angular dimension. Then a multi-scale reconstruction structure is proposed to efficiently implement the non-local attention in the low resolution feature space, while also preserving the high frequency components in the high-resolution feature space. Extensive experiments demonstrate the superior performance of the proposed spatial-angular attention network for reconstructing sparsely-sampled light fields with non-Lambertian effects.

## Results
![Teaser Image](https://gaochangwu.github.io/SAAN/assets/results_large1.png) <br>
Demonstration of attention map on scenes with (a) large disparity and (b) non-Lambertian effect. <br>
![Teaser Image](https://gaochangwu.github.io/SAAN/assets/results_large2.png) <br>
Comparison of the results on the light fields from the CIVIT Dataset (16x upsampling). <br>
![Teaser Image](https://gaochangwu.github.io/SAAN/assets/results_large3.png) <br>
Comparison of the results on the light fields from the MPI Light Field Archive (16x upsampling). <br>


## Note for Code
1. Environment -Python 3.7.4, tensorflow-gpu==1.13.1 <br>

2. You should first upload your light fields to "./Datasets/". <br>

3. The code for 3D light field (1D angular and 2D spatial) reconstruction is "main3d.py". Recommend using the model with upsampling scale \alpha=3 for x8 or x9 reconstruction, and the model with upsampling scale \alpha=4 for x16 reconstruction. <br>

4. Please cite our paper if it helps, thank you! <br>
