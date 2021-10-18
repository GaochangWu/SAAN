- Environment: Tensorflow 1.13.1

- We have two models: "model_SAANx3_shuffle" for x3 upsampling factor and "model_SAANx4_shuffle" for x4 upsampling factor. 

- Recommend use "model_SAANx3_shuffle" for 2x2-8x8 and 3x3-7x7 on Lytro light fields or x9 upsampling on 3D ICME/MPILF light fields.

- Recommend use "model_SAANx4_shuffle" for x16 upsampling on 3D ICME/MPILF light fields.