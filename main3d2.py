import tensorflow as tf
import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt
import utils
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ----------------- Parameters setting -----------------------
sceneName = 'WorkShop'  # Development_dataset_3 Bikes FairyCollection LivingRoom Mannequin WorkShop
dataset = 'MPILF'    # ICME MPILF
sceneFile = '../Datasets/' + dataset + '/'
model_up_scale = 3  # MUST BE 3 or 4
N_svd = 0  # Number of singular values for SVD decomposition. If N_svd=0, we do not perform SVD decomposition.
down_scale = 8
save_img = 0
FLAG_RGB = 1

# ------------------------------------------------------------
up_scale = down_scale
if model_up_scale == 3:
    from model_SAANx3 import model
    modelPath = "./Model/model_SAANx3"
else:
    from model_SAANx4 import model
    modelPath = "./Model/model_SAANx4"
    model_up_scale = 4

result_path = './Results/' + sceneName + 'x' + str(down_scale) + '/'
logWritePath = result_path + 'Log.txt'

num_iter = int(np.ceil(np.log(up_scale) / np.log(model_up_scale)))
pyramid = [2, 3]
if dataset == 'MPILF':
    batch = [24, 24]
else:
    batch = [24, 6]

# -------------- Load light field -----------------
print("Loading light field: %s x%d ... The model is %s" % (sceneName, down_scale, modelPath))

lf_files = glob.glob(sceneFile + sceneName + '/*.png')
ang_ori = len(lf_files)
im = plt.imread(lf_files[0])
[hei, wid, chn] = im.shape
fullLF = np.zeros([hei, wid, chn, ang_ori])
for i in range(0, ang_ori):
    if dataset == 'MPILF':
        cur_im = sceneFile + sceneName + '/Frame_%03d.png' % (i)
    else:
        cur_im = sceneFile + sceneName + '/%04d.png' % (i+1)
    im = plt.imread(cur_im)
    fullLF[:, :, :, i] = im

wid = wid // (np.power(2, pyramid[1] - 1)) * (np.power(2, pyramid[1] - 1))
hei = hei // (np.power(2, pyramid[0] - 1)) * (np.power(2, pyramid[0] - 1))
fullLF = fullLF[0:hei, 0:wid, :, :]
inputLF = fullLF[:, :, :, ::down_scale]

ang_in = inputLF.shape[3]
ang_out = (ang_in-1) * up_scale + 1

utils.mkdir(result_path + 'images/')

with open(logWritePath, 'w') as f:
    f.write("Model name: %s. Input (scene name: %s) is a 1 X %d light field. The output will be a 1 X %d light field.\n" %
            (modelPath, sceneName, ang_in, ang_out))


# ---------------- Model -------------------- #
def slice_reconstruction(size, slice, ang_tar):
    global slice_y
    with sess.as_default():
        slice = utils.rgb2ycbcr(slice)
        if FLAG_RGB:
            slice = np.transpose(slice, (1, 0, 2, 3))
            slice = np.expand_dims(slice, axis=0)

            slice_y = slice[:, :, :, :, 0:1]
            slice_cb = slice[:, :, :, :, 1:2]
            slice_cr = slice[:, :, :, :, 2:3]

            slice_y = sess.run(y_out, feed_dict={x: slice_y})
            slice_cb = sess.run(y_out, feed_dict={x: slice_cb})
            slice_cr = sess.run(y_out, feed_dict={x: slice_cr})

            slice = np.concatenate((slice_y, slice_cb, slice_cr), axis=-1)
            slice = np.transpose(slice[0, :, :, :, :], (1, 0, 2, 3))
            slice = tf.convert_to_tensor(slice)
            slice = tf.image.resize_bicubic(slice, [ang_tar, size])
            slice = slice.eval()
        else:
            slice_y = np.transpose(slice[:, :, :, 0:1], (1, 0, 2, 3))
            slice = tf.convert_to_tensor(slice)
            slice = tf.image.resize_bicubic(slice, [ang_tar, size])
            slice = slice.eval()

            slice_y = np.expand_dims(slice_y, axis=0)
            slice_y = sess.run(y_out, feed_dict={x: slice_y})
            slice_y = tf.convert_to_tensor(np.transpose(slice_y[0], (1, 0, 2, 3)))
            slice_y = tf.image.resize_bicubic(slice_y, [ang_tar, size])
            slice[:, :, :, 0:1] = slice_y.eval()
        slice = utils.ycbcr2rgb(slice)
        slice = np.minimum(np.maximum(slice, 0), 1)
    return slice


# -------------- Light field reconstruction -----------------
print('Reconstructing light field ...')
start = time.time()
global ang_cur_in, lf_in, lf_cur
for i_iter in range(num_iter):
    print('Cascade %d' % (i_iter + 1))
    if i_iter == 0:
        ang_cur_in = ang_in
        lf_in = inputLF
        ang_cur_out = (ang_in - 1) * model_up_scale + 1

    if i_iter == num_iter - 1:
        ang_cur_out = ang_out
    else:
        ang_cur_out = (ang_cur_in - 1) * model_up_scale + 1

    # -------------- Restore graph ----------------
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, ang_cur_in, batch[i_iter], wid, 1])
    y_out = model(x, N_svd)
    g = tf.get_default_graph()
    sess = tf.Session(graph=g)

    with sess.as_default():
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, modelPath)

    # -------------- Reconstruction ----------------
    lf_cur = np.zeros([hei, wid, chn, ang_cur_out])
    n = int(np.ceil(hei/batch[i_iter]))
    for i in range(0, n):
        h_start = i * batch[i_iter]
        if i == n-1:
            h_start = hei - batch[i_iter]
        h_end = (i + 1) * batch[i_iter]
        slice3D = lf_in[h_start:h_end, :, :, :]
        slice3D = np.transpose(slice3D, (0, 3, 1, 2))
        slice3D = slice_reconstruction(wid, slice3D, ang_cur_out)
        lf_cur[h_start:h_end, :, :, :] = np.transpose(slice3D, (0, 2, 3, 1))
    sess.close()

    lf_in = lf_cur
    ang_cur_in = ang_cur_out

out_lf = lf_cur
elapsed = (time.time() - start)
print("Light field reconstruction consumes %.2f seconds, %.3f seconds per view." % (elapsed, elapsed / ang_out))

with open(logWritePath, 'a') as f:
    f.write("Reconstruction completed within %.2f seconds (%.3f seconds averaged on each view).\n"
            % (elapsed, elapsed / ang_out))

# -------------- Evaluation -----------------
psnr = [0 for _ in range(ang_out)]
ssim = [0 for _ in range(ang_out)]
border_cut = 0

for s in range(0, ang_out):
    cur_im = out_lf[:, :, :, s]

    if np.mod(s, up_scale) != 0 and down_scale == up_scale:
        cur_gt = fullLF[:, :, :, s]
        psnr[s], ssim[s] = utils.metric(cur_im, cur_gt, border_cut)

    if save_img:
        plt.imsave(result_path + 'images/' + 'out_' + str(s + 1) + '.png', np.uint8(out_lf[:, :, :, s] * 255))

psnr_avg = np.average(psnr) * ang_out / (ang_out - ang_in)
ssim_avg = np.average(ssim) * ang_out / (ang_out - ang_in)

print("PSNR and SSIM on synthetic views are %2.3f and %1.4f." % (psnr_avg, ssim_avg))
with open(logWritePath, 'a') as f:
    f.write("PSNR and SSIM on synthetic views are %2.3f and %1.4f.\n" % (psnr_avg, ssim_avg))
