import os
import random
import cv2
from setuptools import glob
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

dimension = (98, 98, 3)

data = []
labels = []
# loading the each image path into images_files
images_files = [f for f in glob.glob(r'gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(images_files)

# converting image into array
for img in images_files:
    image = cv2.imread(img)
    new_image = cv2.resize(image, (dimension[0], dimension[1]))
    images = img_to_array(new_image)
    data.append(images)

    label = img.split(os.path.sep)[-2]  # file path is gender_dataset_face\\man\\face_31.jpg
    if label == 'women':
        label = 1
    else:
        label = 0

    labels.append([label])

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# splitting the data into train and test set
train_X, test_X, train_Y, test_Y = train_test_split(data, labels, test_size=0.3)
train_Y = to_categorical(train_Y, num_classes=2)
test_Y = to_categorical(test_Y, num_classes=2)
