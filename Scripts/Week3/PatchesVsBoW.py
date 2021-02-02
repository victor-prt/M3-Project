from utils import *
from keras.models import Sequential, Model, load_model

from keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans

from sklearn.svm import SVC
from PIL import Image
import numpy as np
#user defined variables
PATCH_SIZE  = 32
BATCH_SIZE  = 8
DATASET_DIR = '../MIT_split'
MODEL_FNAME = '../psize_32_bsize_8_layers_4096_2048_1024_8patch_based_mlp.h5'
#MODEL_FNAME = '../my_first_mlp.h5'
PATCHES_DIR = '../MIT_split_patches'


if not os.path.exists(DATASET_DIR):
    colorprint(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
    quit()


print("creador de patches")
colorprint(Color.YELLOW, 'WARNING: patches dataset directory '+PATCHES_DIR+' do not exists!\n')
colorprint(Color.BLUE, 'Creating image patches dataset into '+PATCHES_DIR+'\n')
generate_image_patches_db(DATASET_DIR,PATCHES_DIR,patch_size=PATCH_SIZE)
colorprint(Color.BLUE, 'Done!\n')


colorprint(Color.BLUE, 'Building MLP model...\n')

model = load_model(MODEL_FNAME)

print(model.summary())

colorprint(Color.BLUE, 'Done!\n')

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
print("Creando el train")
train_generator = train_datagen.flow_from_directory(
      PATCHES_DIR+'/train',  # this is the target directory
      target_size=(PATCH_SIZE, PATCH_SIZE),  # all images will be resized to PATCH_SIZExPATCH_SIZE
      batch_size=BATCH_SIZE,
      classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
      class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
      PATCHES_DIR+'/test',
      target_size=(PATCH_SIZE, PATCH_SIZE),
      batch_size=BATCH_SIZE,
      classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
      class_mode='categorical')

categories = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']

Train_descriptors = []
Train_label_per_descriptor = []

#model = load_model(filepath=MODEL_FNAME)
model_layer = Model(inputs=model.input, outputs=model.get_layer('fourth').output)
train_labels=[]
print('Train Labels')
for filename, label in zip(train_generator.filenames, train_generator.classes):
    x = np.asarray(Image.open(os.path.join(PATCHES_DIR+'/train', filename)))
    x = np.expand_dims(Image.fromarray(x).resize((PATCH_SIZE, PATCH_SIZE)), axis=0)
    features = model_layer.predict(x)
    Train_descriptors.append(features[0])
    train_labels.append(label)

size, k= np.shape(Train_descriptors)
visual_words=np.zeros((size,k),dtype=np.float32)

print('Train visual words')
for i in range(len(Train_descriptors)):
    visual_words[i,:]=Train_descriptors[i]

visual_words_test=np.zeros((len(validation_generator.filenames),k),dtype=np.float32)

test_labels=[]
print('Test visual words and labels')
for i in range(len(validation_generator.filenames)):
    x = np.asarray(Image.open(PATCHES_DIR+'/test/' + validation_generator.filenames[i]))
    x = np.expand_dims(Image.fromarray(x).resize((PATCH_SIZE, PATCH_SIZE)), axis=0)
    features = model_layer.predict(x)
    test_labels.append(validation_generator.classes[i])
    visual_words_test[i,:]=features[0]

print('TIME FOR BoW-------------------------------------------')
D=np.vstack(Train_descriptors)
codebook = MiniBatchKMeans(n_clusters=len(categories), verbose=False, batch_size=BATCH_SIZE, compute_labels=False, reassignment_ratio=10 ** -4, random_state=42)
codebook.fit(D)
visual_words=np.zeros((len(Train_descriptors),k),dtype=np.float32)
for i in range(len(Train_descriptors)):
    words=codebook.predict(Train_descriptors[i].reshape(1, -1))
    visual_words[i,:]=np.bincount(words,minlength=k)

knn = KNeighborsClassifier(n_neighbors=8, n_jobs=-1, metric='euclidean')
knn.fit(visual_words, train_labels)

accuracy = knn.score(visual_words_test, test_labels)

print('Done!')
print('-------------------------------------------------------------------------')
print('CNN: ', model.evaluate(validation_generator)[1])
print('BoW: ', accuracy)


