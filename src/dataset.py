import pathlib
import random
import shutil

import tensorflow as tf

from train_config import *

train_water_dir = pathlib.Path(train_water_path)
train_water_paths = list(train_water_dir.glob("*"))
train_cover_dir = pathlib.Path(train_cover_path)
train_cover_paths = list(train_cover_dir.glob("*"))
test_water_dir = pathlib.Path(test_water_path)
test_water_paths = list(test_water_dir.glob("*"))
test_cover_dir = pathlib.Path(test_cover_path)
test_cover_paths = list(test_cover_dir.glob("*"))


def load_water_image(path):
    # 读取图像文件为张量
    image = tf.io.read_file(str(path))
    # 解码为RGB格式
    image = tf.image.decode_jpeg(image, channels=1)
    # 转换为浮点数类型
    image = tf.image.convert_image_dtype(image, tf.float32)
    # 缩放到指定大小
    image = tf.image.resize(image, [water_size, water_size])
    # 使用tf.where函数来对图片进行二值化
    binary_image = tf.where(image > 0.5, 1.0, 0.0)
    # 返回图像张量
    return binary_image


def load_cover_image(path):
    # 读取图像文件为张量
    image = tf.io.read_file(str(path))
    # 解码为RGB格式
    image = tf.image.decode_jpeg(image, channels=3)
    # 转换为浮点数类型
    image = tf.image.convert_image_dtype(image, tf.float32)
    # 缩放到指定大小
    image = tf.image.resize(image, [image_size, image_size])
    # 返回图像张量
    return image


def generate_train():
    while True:
        water_path = random.choice(train_water_paths)
        cover_path = random.choice(train_cover_paths)
        yield load_water_image(water_path), load_cover_image(cover_path)


def generate_test():
    while True:
        water_path = random.choice(test_water_paths)
        cover_path = random.choice(test_cover_paths)
        yield load_water_image(water_path), load_cover_image(cover_path)


train_dataset = tf.data.Dataset.from_generator(generate_train,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=((water_size, water_size, 1), (image_size, image_size, 3))).\
    map(lambda x, y: ((x, y), None)).batch(batch_size=batch_size)
train_dataset.map(lambda x, y: ((x, y), None)).batch(batch_size=batch_size)
test_dataset = tf.data.Dataset.from_generator(generate_test,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=((water_size, water_size, 1), (image_size, image_size, 3))).\
    map(lambda x, y: ((x, y), None)).batch(batch_size=batch_size)


def copy(path, des, count=10000):
    files = os.listdir(path)
    for i in files[0:1000]:
        shutil.copyfile(f"{path}\\{i}", f"{des}\\{i}")


if __name__ == '__main__':
    print(train_dataset.take(1))
