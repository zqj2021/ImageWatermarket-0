import keras.models

import src.callbacks
from loss import *
from src.dataset import train_dataset, test_dataset
from src.loss import dense_loss
from src.train_config import *


def __conv_block(con_input, channel, is_need_loss):
    con_1 = keras.layers.Conv2D(32, (1, 1))(con_input)
    con_2_1 = keras.layers.Conv2D(32, (1, 1))(con_input)
    con_2_2 = keras.layers.Conv2D(32, (3, 3), padding='same')(con_2_1)
    con_3_1 = keras.layers.Conv2D(32, (1, 1))(con_input)
    con_3_2 = keras.layers.Conv2D(32, (3, 3), padding='same')(con_3_1)
    con_3_3 = keras.layers.Conv2D(32, (3, 3), padding='same')(con_3_2)
    concatenation = keras.layers.concatenate([con_1, con_2_2, con_3_3])
    con_out = keras.layers.Conv2D(channel, (1, 1))(concatenation)
    add_out = keras.layers.Add()([con_input, con_out])
    act_out = keras.layers.Activation(activation)(add_out)
    if is_need_loss:
        return [act_out, concatenation, con_out]
    else:
        return act_out


def conv_block(con_input, channel):
    return keras.models.Model(inputs=[con_input], outputs=__conv_block(con_input, channel, False), name=None)(con_input)


def conv_block_with_loss(con_input, channel):
    return keras.models.Model(inputs=[con_input], outputs=__conv_block(con_input, channel, True), name='ConvLoss')


def encoder(input_wi):
    conv1 = conv_block(input_wi, 1)
    conv2 = conv_block(conv1, 1)
    conv_24 = keras.layers.Conv2D(24, (1, 1), activation=activation)(conv2)
    conv3 = conv_block(conv_24, 24)
    conv4 = conv_block(conv3, 24)
    conv_48 = keras.layers.Conv2D(48, (1, 1), activation=activation)(conv4)
    reshape_image = keras.layers.Reshape([128, 128, 3])(conv_48)
    return reshape_image


def embedder(input_water, input_cover):
    concatenate = keras.layers.concatenate([input_water, input_cover])
    # todo 思考<input_cover>未添加激活与激活后的<input_water>拼接是否合理
    conv1 = conv_block(concatenate, 6)
    conv_3 = keras.layers.Conv2D(3, (1, 1), activation='sigmoid')(conv1)
    return conv_3


class FlattenDense(keras.layers.Layer):
    # 初始化函数，定义输出神经元的个数
    def __init__(self, units, name='FlattenDense'):
        super(FlattenDense, self).__init__(name=name)
        self.w = None
        self.units = units

    # 构建函数，定义权重矩阵和偏置向量
    def build(self, input_shape):
        # 定义权重矩阵，使用随机初始化
        self.w = self.add_weight(name='weight', shape=(self.units, input_shape[-1], input_shape[-2], input_shape[-3]),
                                 initializer='random_normal',
                                 trainable=True)

    # 调用函数，定义前向传播的逻辑
    def call(self, inputs):
        channel_first = tf.transpose(inputs, [0, 3, 1, 2])
        exp_i = tf.expand_dims(channel_first, 1)
        mat_res = tf.matmul(self.w, exp_i)
        y_mat = tf.reduce_sum(mat_res, axis=[2])
        loss = dense_loss(y_mat, self.w) * 0.01
        self.add_loss(loss)
        y_out = tf.transpose(y_mat, [0, 2, 3, 1])
        y_activate = keras.activations.relu(y_out)
        return y_activate

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
        })
        return config


def extractor(input_t):
    conv1 = conv_block(input_t, par_N)
    conv2 = conv_block(conv1, par_N)
    conv_1 = keras.layers.Conv2D(3, (1, 1), activation=activation)(conv2)
    out_reshape = keras.layers.Reshape([32, 32, 48])(conv_1)
    return out_reshape


def decoder(input_wi):
    conv1 = conv_block(input_wi, 48)
    conv2 = conv_block(conv1, 48)
    conv_1 = keras.layers.Conv2D(24, (1, 1), activation=activation)(conv2)
    conv3 = conv_block(conv_1, 24)
    conv4 = conv_block(conv3, 24)
    conv_2 = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv4)
    return conv_2


def full_model():
    water_input = keras.layers.Input([water_size, water_size, 1], name='input_water')
    image_input = keras.layers.Input([image_size, image_size, 3], name='input_cover')
    wfi = encoder(water_input)
    loss_model = conv_block_with_loss(wfi, 3)
    model_out, wfi_b1, wfi_b2 = loss_model(wfi)
    mi = embedder(model_out, image_input)
    _, mi_b1, mi_b2 = loss_model(mi)
    Lam3Loss(name='Lamda3Loss')([wfi_b1, wfi_b2, mi_b1, mi_b2])
    _, mi = CoverLoss(name='CoverLoss')([image_input, mi])
    ti = FlattenDense(units=par_N)(mi)
    wfi_ = extractor(ti)
    wi_ = decoder(wfi_)
    _, wi_ = WaterLoss(name='WaterLoss')([water_input, wi_])
    return keras.models.Model(inputs=[water_input, image_input], outputs=[wi_, mi], name='FullModel')


def load_model():
    try:
        _model = tf.keras.models.load_model(model_path)
    except OSError:
        _model = full_model()
        _model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
        print("model is not found. init success!")
    _model.summary()
    return _model


if __name__ == '__main__':
    model = load_model()
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    model.fit(x=train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch,
              callbacks=src.callbacks.callbacks()
              , validation_data=test_dataset, validation_steps=val_steps, initial_epoch=800)
