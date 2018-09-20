from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from keras import backend as keras_backend #for to convert in float32
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import keras
#for to read and show images, better than opencv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#for cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
# for tensorboard
from time import time
from keras.callbacks import TensorBoard
#parameters
Num_epochs = 5
image_height = 128
image_width =128
random_seed = 36
Test_Size = 0.5
number_of_pixels = image_height * image_width
def image_to_feature_vector(image, size=(image_height, image_width)):
    # resize the image to a fixed size, then flatten the image into a list of raw pixel intensities
    return cv2.resize(image, size).flatten()
print("[INFO] describing images...", image_height, image_width)
#imagePaths = list(paths.list_images("/home/javeriana/Dropbox/Javeriana/Courses/RotaciónII-Tamura/Original/All"))
#imagePaths = list(paths.list_images("/home/roger/Dropbox/Javeriana/Courses/RotaciónII-Tamura/Original/All"))
imagePaths = list(paths.list_images("E:\Dropbox\Javeriana\Courses\RotaciónII-Tamura\Original\All"))
data = []
labels = []
for (i, imagePath) in enumerate(imagePaths):
    imagePath=imagePaths[i]
    image = mpimg.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    features = image_to_feature_vector(image)
    data.append(features)
    labels.append(label)
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))
        imgplot = plt.imshow(image)
le = LabelEncoder()     # Encode labels with value between 0 and n_classes-1.
labels = le.fit_transform(labels)
(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=Test_Size, random_state=random_seed)

X_train=np.array(X_train)   #converting to array numpy for obtain shape
X_test = np.array(X_test)
X_train = keras_backend.cast_to_floatx(X_train) #convert to float 32
X_test = keras_backend.cast_to_floatx(X_test)
X_train /= 255.0
X_test /= 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('samples_train shape = ', X_train.shape)
print('samples_test shape = ', X_test.shape)
print('labels_train shape = ', y_train.shape)
print('labels_test shape = ', y_test.shape)

# define the architecture of the network
model = Sequential()
model.add(Dense(768, activation="relu", kernel_initializer="uniform"))
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(2))
model.add(Activation("softmax"))
# train the model using SGD
print("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
print('Test Size = ', Test_Size)
history= model.fit(X_train, y_train, epochs=Num_epochs, batch_size=128,	verbose=1)
# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(X_test, y_test, batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
model.summary()
history.history['acc']