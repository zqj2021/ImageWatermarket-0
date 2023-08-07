import keras.callbacks
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from src.dataset import generate_test
from train_config import *


class ResultCallback(keras.callbacks.Callback):
    def __init__(self, log_dir, generate):
        super().__init__()
        self.gen = generate
        self.image_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        water_o, cover_o = next(self.gen)
        water = tf.expand_dims(water_o, 0)
        cover = tf.expand_dims(cover_o, 0)
        pred_water, pred_cover = self.model.predict((water, cover))

        water_o = tf.image.grayscale_to_rgb(water_o)
        water_o = tf.image.resize(water_o, [128, 128], method=ResizeMethod.NEAREST_NEIGHBOR, antialias=True)

        pred_water = tf.convert_to_tensor(pred_water)
        pred_cover = tf.convert_to_tensor(pred_cover)
        pred_water = tf.squeeze(pred_water, [0])
        pred_cover = tf.squeeze(pred_cover, [0])

        pred_water = tf.image.grayscale_to_rgb(pred_water)
        pred_water = tf.image.resize(pred_water, [128, 128], method=ResizeMethod.NEAREST_NEIGHBOR, antialias=True)
        images = [water_o, cover_o, pred_cover, pred_water]

        images = tf.stack(images)
        with self.image_writer.as_default():
            tf.summary.image(f"Epoch:{epoch}", data=images, step=epoch, max_outputs=4)


def callbacks():
    logdir = log_path
    return [
        keras.callbacks.TensorBoard(write_images=True, log_dir=logdir, histogram_freq=1),
        keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='loss'),
        ResultCallback(log_dir=logdir, generate=generate_test())
    ]
