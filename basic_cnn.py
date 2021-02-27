import numpy as np
import os
import math

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam

from utils import *


batch_size = 16
epochs = 3
num_train = 1000
num_val = 100
num_test = 100
input_shape = (460, 460, 3)

params = {'dim': (460, 460), 'batch_size': batch_size, 'n_channels': 3, 'shuffle': True}


partition = {'train': range(0, num_train), 'validation': range(num_train, num_train+num_val), 'test': range(num_train+num_val, num_train+num_val+num_test)}
labels = np.load('../labels.npy').tolist()
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)
test_generator = DataGenerator(partition['test'], labels, **params)


model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
# model.add(Dense(221, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(3))
model.add(Dense(221, activation='softmax' ))

model.compile(loss=['categorical_crossentropy'], optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)), metrics=['accuracy'])

model.summary()

# model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x,y))
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=math.ceil(num_train / batch_size),
                    validation_steps=math.floor(num_val / batch_size),
                    shuffle=True, epochs=epochs, verbose=1)


model.save('class_basic_cnn.h5')

scores = model.evaluate(test_generator, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

x_test, y_test = get_test_set(10)
x_pred = np.around(model.predict(x_test))
print(x_pred)
print(y_test)
keras.metrics.categorical_crossentropy(x_pred, y_test)
