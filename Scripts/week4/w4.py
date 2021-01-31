
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

print("Checkpoint 1")

train_data_dir='../../Databases/MIT_split/train'
val_data_dir='../../Databases/MIT_split/test'
test_data_dir='../../Databases/MIT_split/test'
img_width = 224
img_height=224
batch_size=32
number_of_epoch=5
validation_samples=807

print("Checkpoint 2")

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_data_format()
    assert dim_ordering in {'channels_first', 'channels_last'}

    if dim_ordering == 'channels_first':
        # 'RGB'->'BGR'
        x = x[ ::-1, :, :]
        # Zero-center by mean pixel
        x[ 0, :, :] -= 103.939
        x[ 1, :, :] -= 116.779
        x[ 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x

print("Checkpoint 3")
# create the base pre-trained model
base_model = MobileNet(weights='imagenet', dropout=0.5)
plot_model(base_model, to_file='modelVGG16a.png', show_shapes=True, show_layer_names=True)

x = base_model.layers[-2].output
x = Dense(8, activation='softmax',name='predictions')(x)

model = Model(inputs=base_model.input, outputs=x)
plot_model(model, to_file='modelVGG16b.png', show_shapes=True, show_layer_names=True)


layers = len(model.layers)
for layer in model.layers:
    layer.trainable = False
for layer in model.layers[layers-2:]:
   layer.trainable = True

print("Checkpoint 4")
    
model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])
print("Checkpoint 5")
for layer in model.layers:
    print(layer.name, layer.trainable)

print("Checkpoint 6")

#preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
	preprocessing_function=preprocess_input,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)

print("Checkpoint 7")

train_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

tbCallBack = TensorBoard(log_dir='./outs/1', histogram_freq=0, write_graph=True)
history=model.fit_generator(train_generator,
        steps_per_epoch=(int(400//batch_size)+1),
        nb_epoch=number_of_epoch,
        validation_data=validation_generator,
        validation_steps= (int(validation_samples//batch_size)+1), callbacks=[tbCallBack])


# result = model.evaluate_generator(test_generator, val_samples=validation_samples)
print("Checkpoint 8")

# print( result)


# list all data in history

print("History: ",history.history)

if True:
  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('accuracy.jpg')
  plt.close()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('loss.jpg')
