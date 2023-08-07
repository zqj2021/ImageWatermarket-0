import datetime

activation = 'relu'
batch_size = 128
par_N = 200
water_size = 32
image_size = 128
par_lam = [1, 1, 1, 0.01]
steps_per_epoch = 10000 / batch_size
val_steps = 20
epochs = 2000

learning_rate = 1e-5
project_path = './'
__save_date_path = project_path + 'save-date/'
model_name = 'DIWatermark_v1.2'
dataset_path = project_path + "dataset/"
log_path = __save_date_path + f"logs/{model_name}/" + datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
model_path = __save_date_path + f"models/{model_name}/"
train_cover_path = dataset_path + 'coco10K'
train_water_path = dataset_path + 'cifar10K'
test_cover_path = dataset_path + 'coco1K'
test_water_path = dataset_path + 'cifar1K'
