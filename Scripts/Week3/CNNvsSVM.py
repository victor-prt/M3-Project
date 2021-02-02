import os
import getpass


from utils import *
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from PIL import Image
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 8
DATASET_DIR = '../MIT_split'
MODEL_FNAME = '../imsize_32_bsize_8_layers_4096_2048_1024_512_8mlp.h5'
#MODEL_FNAME = '../my_first_mlp.h5'

#Build the Multi Layer Perceptron model
model = Sequential()
model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='first'))
model.add(Dense(units=4096, activation='relu',name='second'))
model.add(Dense(units=2048, activation='relu', name='third'))
model.add(Dense(units=1024, activation='relu', name='fourth'))
model.add(Dense(units=512, activation='relu', name='fifth'))

model.add(Dense(units=8, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.load_weights(MODEL_FNAME)
# this is the dataset configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

# this is the dataset configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        DATASET_DIR+'/train',  # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR+'/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')

categories = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']

Train_descriptors = []
Train_label_per_descriptor = []

#model = load_model(filepath=MODEL_FNAME)
model_layer = Model(inputs=model.input, outputs=model.get_layer('fifth').output)
train_labels=[]

for filename, label in zip(train_generator.filenames, train_generator.classes):
    x = np.asarray(Image.open(os.path.join(DATASET_DIR+'/train', filename)))
    x = np.expand_dims(Image.fromarray(x).resize((IMG_SIZE, IMG_SIZE)), axis=0)
    features = model_layer.predict(x)
    Train_descriptors.append(features[0])
    train_labels.append(label)

size, k= np.shape(Train_descriptors)
visual_words=np.zeros((size,k),dtype=np.float32)

for i in range(len(Train_descriptors)):
    visual_words[i,:]=Train_descriptors[i]

visual_words_test=np.zeros((len(validation_generator.filenames),k),dtype=np.float32)

test_labels=[]
for i in range(len(validation_generator.filenames)):
    x = np.asarray(Image.open(DATASET_DIR+'/test/' + validation_generator.filenames[i]))
    x = np.expand_dims(Image.fromarray(x).resize((IMG_SIZE, IMG_SIZE)), axis=0)
    features = model_layer.predict(x)
    test_labels.append(validation_generator.classes[i])
    visual_words_test[i,:]=features[0]

svm = SVC()
svm.fit(visual_words, train_labels)

D=np.vstack(Train_descriptors)


codebook = MiniBatchKMeans(n_clusters=8, verbose=False, batch_size=BATCH_SIZE, compute_labels=False, reassignment_ratio=10 ** -4, random_state=42)
codebook.fit(D)

visual_words=np.zeros((len(Train_descriptors),k),dtype=np.float32)
for i in range(len(Train_descriptors)):
    words=codebook.predict(Train_descriptors[i].reshape(1, -1))
    visual_words[i,:]=np.bincount(words,minlength=k)


knn = KNeighborsClassifier(n_neighbors=8,n_jobs=-1,metric='euclidean')
knn.fit(visual_words, train_labels)

accuracy = knn.score(visual_words_test, test_labels)

print('Done!')
kmeans= KMeans(n_clusters=len(categories))
kmeans.fit(visual_words, train_labels)
print(kmeans.score(visual_words_test, test_labels))
print('Eucliedean: ', accuracy)
print('CNN: ', model.evaluate(validation_generator)[1])
print('SVM: ', svm.score(visual_words_test, test_labels))


