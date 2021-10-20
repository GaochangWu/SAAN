import tensorflow as tf
import numpy as np
import os
from model_SAAN import model
import time
import matplotlib.pyplot as plt
import utils
import glob
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------- Parameters setting -----------------------
sceneFile = 'Lytro/30scenes'  # Lytro/Reflective Lytro/30scenes  Occlusions  LinCamArray  StanfordMicro05  Stanford  Soft3D
scenePath = './Datasets/' + sceneFile + '/'
modelPath = "./Model/model_SAANx3"

results_path = 'Results/' + 'attSAAN/' + sceneFile + 'x7/'
utils.mkdir(results_path)

model_up_scale = 3
FLAG_RGB = 0
tone_coef = 1.0
s_ori, ang_start = 14, 4
t_ori = s_ori
ang_ori = 8
ang_in = 2
ang_out = ang_ori
border_cut = 22
up_scale = int((ang_out - 1) / (ang_in - 1))
num_iter = int(np.ceil(np.log(up_scale) / np.log(model_up_scale)))
pyramid = 4

lf_files = sorted(glob.glob(scenePath + '*.png'))
log_batch_path = results_path + '/Log.txt'
with open(log_batch_path, 'w') as f:
    f.write("Dataset is %s.\n" % sceneFile)

# -------------------------------------------------------------------
psnr_bacth = [0 for _ in range(len(lf_files))]
ssim_bacth = [0 for _ in range(len(lf_files))]
time_bacth = [0 for _ in range(len(lf_files))]
for i in range(0, len(lf_files)):
    sceneName = lf_files[i]
    sceneName = sceneName[len(scenePath):-4]
    cur_path = results_path + sceneName
    log_path = cur_path + '/Log.txt'
    utils.mkdir(cur_path + '/images/')

    # -------------- Load LF -----------------
    print("Loading light field %d of %d: %s..." % (i + 1, len(lf_files), sceneName))
    fullLF, inputLF = utils.load4DLF(scenePath + sceneName, s_ori, t_ori, ang_start, ang_ori, ang_in, tone_coef,
                                     pyramid)
    [hei, wid, chn, s, t] = fullLF.shape
    mean_lf = np.mean(np.mean(inputLF, axis=4), axis=3)
    mean_lf = np.uint8(np.minimum(np.maximum(mean_lf, 0), 1) * 255)

    out_lf = np.zeros([hei, wid, chn, ang_out, ang_out])

    with open(log_path, 'w') as f:
        f.write("Input (scene name: %s) is a %d X %d light field, extracted start from the %d th view. The output will "
                "be a %d X %d light field.\n" % (sceneName, ang_in, ang_in, ang_start, ang_out, ang_out))


    def slice_reconstruction(size, slice, ang_tar):
        # ---------------- Model -------------------- #
        global slice_y
        with sess.as_default():
            if FLAG_RGB:
                # slice_ycbcr = utils.rgb2ycbcr(slice)
                slice = np.transpose(slice, (1, 0, 2, 3))
                slice = np.expand_dims(slice, axis=0)

                slice_y = slice[:, :, :, :, 0:1]
                slice_cb = slice[:, :, :, :, 1:2]
                slice_cr = slice[:, :, :, :, 2:3]

                slice_y = sess.run(y_out, feed_dict={x: slice_y})
                slice_cb = sess.run(y_out, feed_dict={x: slice_cb})
                slice_cr = sess.run(y_out, feed_dict={x: slice_cr})

                slice_ycbcr = np.concatenate((slice_y, slice_cb, slice_cr), axis=-1)
                slice_ycbcr = np.transpose(slice_ycbcr[0, :, :, :, :], (1, 0, 2, 3))
                slice_ycbcr = tf.convert_to_tensor(slice_ycbcr)
                slice_ycbcr = tf.image.resize_bicubic(slice_ycbcr, [ang_tar, size])
                slice = sess.run(slice_ycbcr)
                # slice = utils.ycbcr2rgb(slice_ycbcr)
            else:
                slice_ycbcr = utils.rgb2ycbcr(slice)
                slice_y = np.transpose(slice_ycbcr[:, :, :, 0:1], (1, 0, 2, 3))

                slice_ycbcr = tf.convert_to_tensor(slice_ycbcr)
                slice_ycbcr = tf.image.resize_bicubic(slice_ycbcr, [ang_tar, size])
                slice_ycbcr = sess.run(slice_ycbcr)
                
                slice_y = np.expand_dims(slice_y, axis=0)
                slice_y = sess.run(y_out, feed_dict={x: slice_y})
                slice_y = tf.convert_to_tensor(np.transpose(slice_y[0], (1, 0, 2, 3)))
                slice_y = tf.image.resize_bicubic(slice_y, [ang_tar, size])
                slice_ycbcr[:, :, :, 0:1] = sess.run(slice_y)
                slice = utils.ycbcr2rgb(slice_ycbcr)
            slice = np.minimum(np.maximum(slice, 0), 1)
        return slice


    # -------------- Column reconstruction -----------------
    start1 = time.time()
    global ang_cur_in, lf_in
    for s in range(0, ang_in):
        cur_s = s * up_scale
        slice3D = inputLF[:, :, :, s, :]

        for i_iter in range(num_iter):
            if i_iter == 0:
                ang_cur_in = ang_in
                lf_in = slice3D
                ang_cur_out = (ang_in - 1) * model_up_scale + 1

            if i_iter == num_iter - 1:
                ang_cur_out = ang_out
            else:
                ang_cur_out = (ang_cur_in - 1) * model_up_scale + 1

            # -------------- Restore graph ----------------
            x = tf.placeholder(tf.float32, shape=[None, ang_cur_in, hei, wid, 1])
            y_out = model(x)
            g = tf.get_default_graph()
            sess = tf.Session(graph=g)

            with sess.as_default():
                saver = tf.train.Saver(tf.global_variables())
                saver.restore(sess, modelPath)

            lf_in = np.transpose(lf_in, (0, 3, 1, 2))
            lf_in = slice_reconstruction(wid, lf_in, ang_cur_out)
            lf_in = np.transpose(lf_in, (0, 2, 3, 1))
            ang_cur_in = ang_cur_out

            tf.reset_default_graph()
            sess.close()
        out_lf[:, :, :, cur_s:cur_s + 1, :] = np.expand_dims(lf_in, axis=3)
    elapsed1 = (time.time() - start1)

    # -------------- Row reconstruction -----------------
    start2 = time.time()
    for t in range(0, ang_out):
        if np.mod(t, up_scale) == 0:
            slice3D = inputLF[:, :, :, :, int(t / up_scale)]
        else:
            slice3D = out_lf[:, :, :, ::up_scale, t]

        for i_iter in range(num_iter):
            if i_iter == 0:
                ang_cur_in = ang_in
                lf_in = slice3D
                ang_cur_out = (ang_in - 1) * model_up_scale + 1

            if i_iter == num_iter - 1:
                ang_cur_out = ang_out
            else:
                ang_cur_out = (ang_cur_in - 1) * model_up_scale + 1

            # -------------- Restore graph ----------------
            x = tf.placeholder(tf.float32, shape=[None, ang_cur_in, wid, hei, 1])
            y_out = model(x)
            g = tf.get_default_graph()
            sess = tf.Session(graph=g)

            with sess.as_default():
                saver = tf.train.Saver(tf.global_variables())
                saver.restore(sess, modelPath)

            lf_in = np.transpose(lf_in, (1, 3, 0, 2))
            lf_in = slice_reconstruction(hei, lf_in, ang_cur_out)
            lf_in = np.transpose(lf_in, (2, 0, 3, 1))
            ang_cur_in = ang_cur_out
            tf.reset_default_graph()
            sess.close()
        out_lf[:, :, :, :, t:t + 1] = np.expand_dims(lf_in, axis=4)
    elapsed2 = (time.time() - start1)

    with open(log_path, 'a') as f:
        f.write("Reconstruction completed within %.2f seconds (%.3f seconds averaged on each view).\n"
                % (elapsed1 + elapsed2, (elapsed1 + elapsed2) / (ang_out * ang_out)))

    # -------------- Evaluation -----------------
    psnr = np.zeros([ang_out, ang_out])
    ssim = np.zeros([ang_out, ang_out])

    for s in range(0, ang_out):
        for t in range(0, ang_out):
            cur_im = out_lf[:, :, :, s, t]

            if np.mod(s, up_scale) != 0 or np.mod(t, up_scale) != 0:
                if ang_out == ang_ori:
                    cur_gt = fullLF[:, :, :, s, t]
                    psnr[s, t], ssim[s, t] = utils.metric(cur_im, cur_gt, border_cut)

            plt.imsave(cur_path + '/images/' + 'out_' + str(s + 1) + '_' + str(t + 1) + '.png',
                       np.uint8(out_lf[:, :, :, s, t] * 255))

    psnr_avg = np.average(psnr) * ang_out * ang_out / (ang_out * ang_out - ang_in * ang_in)
    ssim_avg = np.average(ssim) * ang_out * ang_out / (ang_out * ang_out - ang_in * ang_in)
    psnr_bacth[i] = psnr_avg
    ssim_bacth[i] = ssim_avg
    time_bacth[i] = elapsed1 + elapsed2

    print("PSNR and SSIM on synthetic views are %2.3f and %1.4f." % (psnr_avg, ssim_avg))
    with open(log_path, 'a') as f:
        f.write("PSNR and SSIM on synthetic views are %2.3f and %1.4f.\n" % (psnr_avg, ssim_avg))
    with open(log_batch_path, 'a') as f:
        f.write("%s: %2.3f, %1.4f.\n" % (sceneName, psnr_avg, ssim_avg))

print("PSNR and SSIM on the dataset are %2.3f and %1.4f. Time cosuming: %.2f seconds per light field" %
      (np.average(psnr_bacth), np.average(ssim_bacth), np.average(time_bacth)))
with open(log_batch_path, 'a') as f:
    f.write("PSNR and SSIM on the dataset are %2.3f and %1.4f. Time cosuming: %.2f seconds per light field\n" %
            (np.average(psnr_bacth), np.average(ssim_bacth), np.average(time_bacth)))
