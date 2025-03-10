import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

path = "digits"

myList = os.listdir(path)
num_of_classes = len(myList)

print("Label(sınıf) sayısı: ", num_of_classes)

images = []
classNo = []

for i in range(num_of_classes):
    my_img_list = os.listdir(path + "\\" + str(i))
    
    for j in my_img_list:
        img = cv2.imread(path + "\\" + str(i) + "\\" + j)
        
        img = cv2.resize(img, (32,32))
        
        images.append(img)
        classNo.append(i)
        
print(len(images)) 
print(len(classNo))

images = np.array(images)
classNo = np.array(classNo)

print(images.shape)
print(classNo.shape)       

train_X, test_X, train_y, test_y = train_test_split(images, classNo, test_size=0.5, random_state=42) 
train_X, validation_X, train_y, validation_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

print(images.shape)
print(train_X.shape)
print(test_X.shape)
print(validation_X.shape)

# gorsellestirme

"""fig, axes = plt.subplots(3, 1, figsize=(7,7))
fig.subplots_adjust(hspace=0.5)  # Hata vardı, `subplot` yerine `subplots_adjust` olmalı!

sns.countplot(x=train_y.astype(str), ax=axes[0])
axes[0].set_title("Y train")

sns.countplot(x=test_y.astype(str), ax=axes[1])
axes[1].set_title("Y test")

sns.countplot(x=validation_y.astype(str), ax=axes[2])
axes[2].set_title("Y validation")

plt.show()"""

# Preprocessing

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255

    return img

# idx = 50
# img = preprocess(train_X[idx])
# img = cv2.resize(img, (300, 300))
# cv2.imshow("Preprocess", img)    

train_X = np.array(list(map(preprocess, train_X)))
test_X = np.array(list(map(preprocess, test_X)))
validation_X = np.array(list(map(preprocess, validation_X)))


train_X = train_X.reshape(-1, 32, 32, 1)
test_X = test_X.reshape(-1, 32,32, 1)
validation_X = validation_X.reshape(-1, 32, 32, 1)

#datagen
dataGen = ImageDataGenerator(width_shift_range = 0.1, 
                                   height_shift_range = 0.1,
                                   zoom_range = 0.1,
                                   rotation_range = 10)


dataGen.fit(train_X)

train_y = to_categorical(train_y, num_of_classes)
test_y = to_categorical(test_y, num_of_classes)
validation_y = to_categorical(validation_y, num_of_classes)


model = Sequential()
model.add(Conv2D(input_shape = (32,32,1), filters = 8, kernel_size = (5,5), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(filters = 16, kernel_size = (3,3), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(units=256, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(units = num_of_classes, activation = "softmax"))

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

batch_size = 250
hist = model.fit(dataGen.flow(train_X, train_y, batch_size = batch_size), validation_data = (validation_X, validation_y), epochs = 15, steps_per_epoch = train_X.shape[0]//batch_size, shuffle = True)             

pickle_out = open("model_trained_new.p","wb")
pickle.dump(model, pickle_out)
pickle_out.close()

#%% degerlendir

hist.history.keys()

plt.figure()
plt.plot(hist.history["loss"], label = "Eğitim Kaybi")
plt.plot(hist.history["val_loss"], label = "validation Kaybi")
plt.legend()
plt.show()


plt.figure()
plt.plot(hist.history["accuracy"], label = "Eğitim accuracy")
plt.plot(hist.history["val_accuracy"], label = "validation accuracy")
plt.legend()
plt.show()

score = model.evaluate(test_X, test_y, verbose = 1)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])


y_pred = model.predict(validation_X)
y_pred_class =  np.argmax(y_pred, axis = 1)
y_true = np.argmax(validation_y, axis = 1)

cm = confusion_matrix(y_true, y_pred_class)

print(cm)

f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm, annot = True, lineWidths = 0.01, cmap = "Greens", linecolor = "gray", fmt = "d", ax = ax)
plt.xlabel("predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()








