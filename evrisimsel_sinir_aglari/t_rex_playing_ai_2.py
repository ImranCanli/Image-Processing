import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

#warnings.filterwarnings("ignore") --> Bu kısım uyarıları kapatmak icin ama ben kapatmicam

images = glob.glob("./img/*.png")

width = 125
height = 50

X = []
Y = []

for img in images:
    
    filename = os.path.basename(img)
    label = filename.split("_")[0]
    
    im =  np.array(Image.open(img).convert("L").resize((width, height))) 
    im = im/255
    X.append(im)
    Y.append(label)

X = np.array(X)
X = X.reshape(X.shape[0], width, height, 1)
                     
#sns.countplot(Y)

def onehot_labels(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    
    one_hot_encoder = OneHotEncoder(sparse_output = False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    
    one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
    
    return one_hot_encoded

Y = onehot_labels(Y)
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size = 0.25, random_state=2)

#cnn model
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (width, height, 1)))
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(3, activation = "softmax"))


"""if os.path.exists("./trex_weight.h5"):
    model.load_weights("trex.weight.h5")
    print("Weight'ler yuklendi")"""

model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])

model.fit(train_X, train_y, epochs = 35, batch_size = 64)

score_train = model.evaluate(train_X, train_y)
print("egitim dogrulugu: %", score_train[1]*100)

score_test = model.evaluate(test_X, test_y)
print("Test dogrulugu: %", score_test[1]*100)

open("model_new.json", "w").write(model.to_json())
model.save_weights("trex_weights_new.weights.h5")
