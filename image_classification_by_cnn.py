# importing libs
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import zipfile
import cv2
import tensorflow as tf
from google.colab.patches import cv2_imshow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator # to read the images without for loop

tf.__version__

# loading the images
from google.colab import drive
drive.mount('/content/drive')

path = '/content/drive/MyDrive/Computer Vision/Datasets/homer_bart_2.zip'
zip_object = zipfile.ZipFile(file=path, mode='r')
zip_object.extractall('./')
zip_object.close()

# tensorflow will use the name of each folder to set each of the classes

tf.keras.preprocessing.image.load_img('/content/homer_bart_2/training_set/bart/bart105.bmp')

tf.keras.preprocessing.image.load_img('/content/homer_bart_2/training_set/homer/homer102.bmp')

# train/ test set

training_generator = ImageDataGenerator(rescale=1./255, # rescale=1./255 is a way to normalize data,
                                        rotation_range=7, # we can also augument data by the use of this func.
                                        horizontal_flip=True,
                                        zoom_range=0.2)

train_dataset = training_generator.flow_from_directory('/content/homer_bart_2/training_set',  # flow_from_directory to read the whole file
                                                        target_size=(64, 64),
                                                        batch_size=8,   # using mini_batch here
                                                        class_mode='categorical',  # if categrical -> 2 output node, elif binary -> 1 output node
                                                        shuffle=True # so the NN wouldn't learn the order of images
                                                       )

train_dataset.classes

train_dataset.class_indices

test_generator = ImageDataGenerator(rescale=1./255)
test_dataset = test_generator.flow_from_directory('/content/homer_bart_2/test_set',
                                                  target_size=(64, 64),
                                                  batch_size=1,   # testing 1 picture after another
                                                  class_mode='categorical',
                                                  shuffle=False)  # if shuffle true while test, we won't be able to associate the predictions with expected outputs(very imp)

# building and training the NN

network = Sequential()  # NN is a sequence of layers & units
network.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3))) # 32 for filter number is kinda a default value, input shape must be the same as train & test size
network.add(MaxPool2D(pool_size=(2, 2)))

network.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu') )
network.add(MaxPool2D(pool_size=(2, 2)))

network.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu') )
network.add(MaxPool2D(pool_size=(2, 2)))

network.add(Flatten())  # no parameter is needed here it only converts the matrix to a vector

network.add(Dense(units=577, activation='relu'))  # hidden layer
network.add(Dense(units=577, activation='relu'))  # hidden layer
network.add(Dense(units=2, activation='softmax'))  # output layer

network.summary()

(1152 + 2) / 2

network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])

history = network.fit(train_dataset, epochs=50)

# evaluating the NN

test_dataset.class_indices

predictions = network.predict(test_dataset)
predictions

# extracting the max value among pred values for softmax alg

predictions = np.argmax(predictions, axis=1)  # axis=1 cuz we wanna access the column
predictions

test_dataset.classes  # expected outputs

from sklearn.metrics import accuracy_score
accuracy_score(test_dataset.classes, predictions)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_dataset.classes, predictions)

sns.heatmap(cm, annot=True);

from sklearn.metrics import classification_report
print(classification_report(test_dataset.classes, predictions))

# saving & loading the model

model_json = network.to_json()
with open('network.json', 'w') as json_file:
  json_file.write(model_json)

from keras.models import  save_model
network_saved = save_model(network, '/content/weights.hdf5')

with open('network.json', 'r') as json_file:
  json_saved_model = json_file.read()
json_saved_model

network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights('weights.hdf5')
network_loaded.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

network_loaded.summary()

# classifying one single image

image = cv2.imread('/content/homer_bart_2/test_set/homer/homer15.bmp')

cv2_imshow(image)

image.shape

image = cv2.resize(image, (64, 64))   # working with higher dimensions will get better results cuz we lose info in small pics, but it would take longer
cv2_imshow(image)

# normalizing the image data

image = image / 255
image

image.shape

image = image.reshape(-1, 64, 64, 3)  # adding another dimension of "1" to send the pics in batch format, for example if we wanna send 10 images we can change the 1 to 10
image.shape

result = network_loaded.predict(image)
result

result = np.argmax(result)
result

test_dataset.class_indices

if result == 0:
  print('Bart')
else:
  print('Homer')