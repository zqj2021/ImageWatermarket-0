import tensorflow as tf
from keras.api import keras


def psnr(true, pred):
    return tf.image.psnr(true, pred, max_val=1.0)


def ber(y_true, y_pred):
    threshold = 0.5  # 阈值
    y_pred = tf.cast(tf.greater(y_pred, threshold), tf.float32)  # 对预测值进行二值化处理
    errors = tf.cast(tf.not_equal(y_true, y_pred), tf.float32)  # 计算错误的比特数
    total_bits = tf.cast(tf.size(y_true[0]), tf.float32)  # 总比特数
    return tf.reduce_sum(errors, axis=[1, 2, 3]) / total_bits  # 计算误码率


class WaterLoss(keras.layers.Layer):
    def __int__(self):
        super(WaterLoss, self).__init__()

    def call(self, inputs):
        water_true, water_pred = inputs
        # self.add_loss(tf.reduce_mean(tf.abs(water_true - water_pred), axis=[1, 2, 3])
        self.add_loss(keras.losses.binary_crossentropy(water_true, water_pred, axis=[1, 2, 3]))
        self.add_metric(ber(water_true, water_pred), name='Ber')
        return inputs


class CoverLoss(keras.layers.Layer):
    def __int__(self):
        super(CoverLoss, self).__init__()

    def call(self, inputs):
        cover_true, cover_pred = inputs
        self.add_loss(tf.reduce_mean(tf.abs(cover_true - cover_pred), axis=[1, 2, 3]))
        self.add_metric(psnr(cover_true, cover_pred), name='Psnr')
        return inputs


def gram_matrix(x):
    x = keras.layers.Reshape([-1, x.shape[-1]])(x)
    gram = tf.matmul(x, x, transpose_b=True)
    return gram


class Lam3Loss(keras.layers.Layer):
    def __int__(self):
        super(WaterLoss, self).__init__()

    def call(self, inputs):
        wfi_b1, wfi_b2, mi_b1, mi_b2 = inputs
        gam_b1 = gram_matrix(wfi_b1) - gram_matrix(mi_b1)
        l1_b1 = tf.reduce_mean(gam_b1, axis=[1, 2])
        gam_b2 = gram_matrix(wfi_b2) - gram_matrix(mi_b2)
        l1_b2 = tf.reduce_mean(gam_b2, axis=[1, 2])
        self.add_loss(0.5 * (l1_b1 + l1_b2))
        return inputs


def dense_loss(h, w):
    h_simple = tf.reduce_mean(h, axis=[-2, -1])
    w_simple = tf.reduce_mean(w, axis=[-2, -1])
    t1 = tf.pow(1 - tf.pow(h_simple, 2), 2)
    wp_simple = tf.pow(w_simple, 2)
    t_sum = tf.reduce_sum(wp_simple, axis=[-1])
    t1 = t1 * t_sum
    t_loss = tf.reduce_sum(t1, axis=[-1])
    return t_loss
