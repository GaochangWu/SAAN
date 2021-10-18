import tensorflow as tf
from tensorflow.keras.layers import Cropping3D
from tensorflow.layers import Conv3D, Conv3DTranspose
# SAAN with deconv inside + split conv + Unet plus backbone


def beta_variable(shape, scope=None, name=None):
    with tf.variable_scope(name_or_scope=scope, default_name=name):
        initial = tf.constant(0.0, shape=shape, dtype=tf.float32, name=name)
        return tf.Variable(initial)


def pixel_shuffle(x, up_scale):
    bsize, ang, hei, wid, chn = x.get_shape().as_list()
    H = tf.reshape(x, [-1, ang, hei, wid, up_scale, chn // up_scale])
    H = tf.split(H, ang, 1)  # ang * (b, 1, hei, wid, 3, chn // up_scale)
    H = tf.concat([tf.squeeze(h, axis=1) for h in H], axis=3)  # (b, hei, wid, ang * up_scale, chn // up_scale)
    return tf.transpose(H, [0, 3, 1, 2, 4])


def SAAM(x, up_scale, N_svd):
    bsize, ang, hei, wid, chn = x.get_shape().as_list()
    # \phi'_q
    h1 = Conv3D(chn // 8, (1, 1, 1), use_bias=False, padding='SAME', name='attn_epi_1')(x)
    h1 = tf.transpose(h1, [0, 2, 1, 3, 4])
    h1 = tf.reshape(h1, [-1, ang * wid, chn // 8])
    if N_svd > 0:
        s1, u1, v1 = tf.svd(h1)
        s1 = tf.slice(s1, [0, 0], [-1, N_svd])
        u1 = tf.slice(u1, [0, 0, 0], [-1, -1, N_svd])
        v1 = tf.slice(v1, [0, 0, 0], [-1, -1, N_svd])

    # \phi'_k
    h2 = Conv3D(chn // 8, (1, 1, 1), use_bias=False, padding='SAME', name='attn_epi_2')(x)
    h2 = tf.transpose(h2, [0, 2, 1, 3, 4])
    h2 = tf.reshape(h2, [-1, ang * wid, chn // 8])
    if N_svd > 0:
        s2, u2, v2 = tf.svd(h2)
        s2 = tf.slice(s2, [0, 0], [-1, N_svd])
        u2 = tf.slice(u2, [0, 0, 0], [-1, -1, N_svd])
        v2 = tf.slice(v2, [0, 0, 0], [-1, -1, N_svd])

    # \phi'_v
    h3 = Conv3D(chn // 8 * up_scale, (1, 1, 1), use_bias=False, padding='SAME', name='attn_epi_3')(x)
    h3 = tf.transpose(h3, [0, 2, 1, 3, 4])
    h3 = tf.reshape(h3, [-1, ang * wid, chn // 8 * up_scale])

    # Map
    if N_svd > 0:
        attn_EPI = tf.matmul(tf.matmul(u1, tf.matmul(tf.matmul(tf.matrix_diag(s1), tf.matmul(v1, v2, transpose_a=True)),
                                                     tf.matrix_diag(s2), transpose_b=True)), u2, transpose_b=True)
    else:
        attn_EPI = tf.matmul(h1, h2, transpose_b=True)
    attn_EPI = tf.nn.softmax(attn_EPI)

    # \phi'_a
    h = tf.matmul(attn_EPI, h3)

    # \phi'_b
    h = tf.reshape(h, [-1, hei, ang, wid, chn // 8 * up_scale])
    h = tf.transpose(h, [0, 2, 1, 3, 4])
    
    x = Conv3D(chn // 8 * up_scale, (1, 1, 1), use_bias=False, padding='SAME', name='attn_epi_4')(x)
    sigma = beta_variable([1], name='attn_sigma')
    h = x + sigma * h

    # \phi_c
    h = pixel_shuffle(h, up_scale)
    h = Cropping3D(cropping=((0, up_scale - 1), (0, 0), (0, 0)))(h)
    h = Conv3D(filters=chn, kernel_size=(1, 1, 7), activation='relu', padding='SAME', name='epi_5')(h)
    h = Conv3D(filters=chn, kernel_size=(7, 1, 1), activation='relu', padding='SAME', name='epi_6')(h)
    return h


def model(x, N_svd=0):
    up_scale = 3
    with tf.variable_scope('ASR'):
        input_shape = x.get_shape().as_list()
        chn_in = input_shape[4]
        chn_base = 24
        # shape is [batch, 6, 24, 64, 1]

        # Group 1
        h = Conv3D(filters=chn_base, kernel_size=(3, 1, 3), activation='relu', padding='SAME', name='conv1_1')(x)
        h = Conv3D(filters=chn_base, kernel_size=(3, 3, 1), activation='relu', padding='SAME', name='conv1_2')(h)
        h1 = Conv3DTranspose(chn_base, (7, 1, 3), (up_scale, 1, 1), 'SAME', activation='relu', name='deconv1')(h)
        h1 = Cropping3D(cropping=((0, up_scale - 1), (0, 0), (0, 0)))(h1)
        h1 = Conv3D(filters=chn_base, kernel_size=(1, 1, 1), activation='relu', padding='SAME', name='conv1_3')(h1)
        # shape is [batch, 6, 24, 64, chn_base]
        h = Conv3D(filters=chn_base * 2, kernel_size=(1, 3, 3), strides=(1, 2, 2), activation='relu', padding='SAME',
                   name='conv1_4')(h)
        # shape is [batch, 6, 12, 32, chn_base * 2]

        # Group 2
        h = Conv3D(filters=chn_base * 2, kernel_size=(3, 1, 3), activation='relu', padding='SAME', name='conv2_1')(h)
        h = Conv3D(filters=chn_base * 2, kernel_size=(3, 3, 1), activation='relu', padding='SAME', name='conv2_2')(h)
        h2 = Conv3DTranspose(chn_base, (7, 1, 3), (up_scale, 1, 1), 'SAME', activation='relu', name='deconv2')(h)
        h2 = Cropping3D(cropping=((0, up_scale - 1), (0, 0), (0, 0)))(h2)
        h2 = Conv3D(filters=chn_base * 2, kernel_size=(1, 1, 1), activation='relu', padding='SAME', name='conv2_3')(h2)
        # shape is [batch, 6, 12, 32, chn_base * 2]
        h = Conv3D(filters=chn_base * 4, kernel_size=(1, 1, 3), strides=(1, 1, 2), activation='relu', padding='SAME',
                   name='conv2_4')(h)
        # shape is [batch, 6, 12, 16, chn_base * 4]

        # Layer 3, shrinking
        h = Conv3D(filters=chn_base * 2, kernel_size=(1, 1, 1), activation='relu', padding='SAME', name='conv3')(h)
        # shape is [batch, 6, 12, 16, chn_base * 2]

        # Group 4, Mapping
        for i in range(2):
            h = Conv3D(filters=chn_base * 2, kernel_size=(3, 1, 3), activation='relu', padding='SAME',
                       name='conv4_1_' + str(i))(h)
            h = Conv3D(filters=chn_base * 2, kernel_size=(3, 3, 1), activation='relu', padding='SAME',
                       name='conv4_2_' + str(i))(h)
        # shape is [batch, 6, 12, 16, chn_base * 2]

        # Group 5, Attention
        h = SAAM(h, up_scale=up_scale, N_svd=N_svd)

        # Layer 6, Expanding
        h = Conv3D(filters=chn_base * 4, kernel_size=(1, 1, 1), activation='relu', padding='SAME', name='conv6')(h)
        # shape is [batch, 6, 12, 16, chn_base * 4]

        # Group 7
        h = Conv3DTranspose(chn_base * 2, (1, 1, 4), (1, 1, 2), activation='relu', padding='SAME', name='conv7_1')(h)
        # shape is [batch, 16, 12, 32, chn_base * 2]
        h = tf.concat([h, h2], axis=-1)
        h = Conv3D(filters=chn_base * 2, kernel_size=(3, 1, 3), activation='relu', padding='SAME', name='conv7_2')(h)
        h = Conv3D(filters=chn_base * 2, kernel_size=(3, 3, 1), activation='relu', padding='SAME', name='conv7_3')(h)
        # shape is [batch, 16, 12, 32, chn_base * 2]

        # Group 8
        h = Conv3DTranspose(chn_base, (1, 4, 4), (1, 2, 2), activation='relu', padding='SAME', name='conv8_1')(h)
        # shape is [batch, 16, 24, 64, chn_base]
        h = tf.concat([h, h1], axis=-1)
        h = Conv3D(filters=chn_base, kernel_size=(3, 1, 3), activation='relu', padding='SAME', name='conv8_2')(h)
        h = Conv3D(filters=chn_base, kernel_size=(3, 3, 1), activation='relu', padding='SAME', name='conv8_3')(h)
        # shape is [batch, 16, 24, 64, chn_base]

        # Group 9
        h = Conv3D(filters=chn_in, kernel_size=(3, 3, 3), padding='SAME', name='conv9')(h)
    return h
