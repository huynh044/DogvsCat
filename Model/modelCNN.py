import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd

data_path = "D:\All_learn_programs\Python\ML_and_DL\Test\DogvsCat\Data\dog_vs_cat_data.csv"
df = pd.read_csv(data_path)
IMAGE_SIZE = 32
X = np.array([np.array(eval(row)) for row in df['pixel_data']])
Y = np.array(df['label'])

X_train_val,X_test,Y_train_val,Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train,Y_val = train_test_split(X_train_val,Y_train_val, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1, 50,50,1)
X_test = X_test.reshape(-1,50,50,1)
X_val = X_val.reshape(-1,50,50,1)

y_train = to_categorical(Y_train, 2)
y_val = to_categorical(Y_val, 2)
y_test = to_categorical(Y_test, 2)

model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(50,50,1), activation="relu"))
model.add(Conv2D(32,(3,3), activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2, activation="softmax"))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
H = model.fit(X_train, y_train, validation_data=(X_val, y_val),
              batch_size=32, epochs=10, verbose=1)
model.save("model_cnn.h5")
fig = plt.figure()
numOfEpoch = 10
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='Training Loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='Validation Loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='Training Accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print(score)




