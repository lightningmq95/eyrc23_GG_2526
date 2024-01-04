# %%
import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

# %%
folder_path = 'GG_4A'

# %%
train_dir = os.path.join(folder_path, 'train')
test_dir = os.path.join(folder_path, 'test')

# %%
img_height,img_width=224,224
batch_size=32
train_ds = keras.preprocessing.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = keras.preprocessing.image_dataset_from_directory(
  test_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

resnet_model = Sequential()

pretrained_model= keras.applications.efficientnet.EfficientNetB0(include_top=False,
  input_shape=(224,224,3),
  pooling='avg',classes=5,
  weights='imagenet')

for layer in pretrained_model.layers:
  layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(256, activation='relu'))
resnet_model.add(Dense(5, activation='softmax'))
resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

epochs=10
history = resnet_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

resnet_model.save('keras_model.h5')

# %%



