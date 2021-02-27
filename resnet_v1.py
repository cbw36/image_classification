from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import add
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
import numpy as np
import os


"""
layers = # of convolutions. pooling, BN, activation, etc. don't count
block: a group of layers on which the residual learning takes place
stage: a collection of blocks with equal filter size
"""


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


#subtract per pixel mean
subtract_pixel_mean = True
if subtract_pixel_mean:
    train_images_mean = np.mean(train_images, axis=0)
    train_images -= train_images_mean
    test_images -= train_images_mean

#account for channel dimension=1 or 3


#data augmentation

batch_size = 32  # orig paper trained all networks with batch_size=128 for CIFAR and 256 for Imagenet
num_classes=10
kernel_size=(3, 3)
strides=(1,1)
NUM_LAYERS_PER_BLOCK = 2
NUM_BLOCKS_PER_STAGE = 3  # number of residual blocks per feature map size. called n in paper
NUM_LAYERS = 6*NUM_BLOCKS_PER_STAGE + 2
NUM_STAGES = 3
FILTERS = (16, 32, 64)
regularizer = l2(0.0001)
momentum = 0.9
epochs = 10
activation="relu"


# Convert class vectors to binary class matrices.
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def resnet(input):
    x = input
    x = Conv2D(filters=FILTERS[0],
               kernel_size=(3, 3),
               strides=(1,1),
               padding="same",
               kernel_regularizer=regularizer,
               kernel_initializer='he_normal',
               name='conv_init')(x)
    x = BatchNormalization(name="bn_conv_init")(x)  # TODO required?
    x = Activation('relu')(x)

    for stage in range(NUM_STAGES):
        for block in range(NUM_BLOCKS_PER_STAGE):
            if (stage == 0) or (block != 0):
                x = identityBlock(x, FILTERS[stage], stage, block)
            else:
                x = downsampleBlock(x, FILTERS[stage], stage, block)
        # if stage==0:
        #     x = identityBlock(x, FILTERS[stage], stage+1, 1)
        # else:
        #     x = downsampleBlock(x, FILTERS[stage], stage+1, 1)
        # x = identityBlock(x, FILTERS[stage], stage+1, 2)
        # x = identityBlock(x, FILTERS[stage], stage+1, 3)

    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    # Instantiate model.
    model = Model(inputs=input, outputs=outputs)
    return model

def convolutionLayer(filter, kernel_size=(3, 3), strides=(1,1), padding="same",
         kernel_regularizer=l2(0.0001), kernel_initializer='he_normal'):
    return Conv2D(filters=filter,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding=padding,
                  kernel_regularizer=kernel_regularizer,
                  kernel_initializer=kernel_initializer)


def identityBlock(input, filter, stage, block):
    conv_name_base = 'conv' + str(stage) + '_' + str(block) + '_layer_'
    bn_name_base = 'bn' + str(stage) + '_' + str(block) + '_layer_'

    x = Conv2D(filters=filter, kernel_size=3, strides=1, padding='same', name=conv_name_base + '1')(input)
    x = BatchNormalization(name=bn_name_base + '1')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filter, kernel_size=3, strides=1, padding='same', name=conv_name_base + '2')(x)
    x = BatchNormalization(name=bn_name_base + '2')(x)

    x = add([x, input])
    x = Activation('relu')(x)

    return x


def downsampleBlock(input, filter, stage, block):
    conv_name_base = 'conv' + str(stage) + '_' + str(block) + '_layer_'
    bn_name_base = 'bn' + str(stage) + '_' + str(block) + '_layer_'

    x = Conv2D(filters=filter, kernel_size=3, strides=2, padding='same', name=conv_name_base + '1')(input)
    x = BatchNormalization(name=bn_name_base + '1')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filter, kernel_size=3, strides=1, padding='same', name=conv_name_base + '2')(x)
    x = BatchNormalization(name=bn_name_base + '2')(x)

    input = Conv2D(filters=filter, kernel_size=1, strides=2, padding='same', name=conv_name_base + 'skip_connection')(input)
    # input = BatchNormalization(name=bn_name_base + 'skip_connection')(input)  # TODO required?

    x = add([x, input])
    x = Activation('relu')(x)

    return x



input = Input(shape=train_images.shape[1:])
x = input
model = resnet(input)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()



# Prepare model model saving directory.
model_name = 'ResNet%dv1' % (NUM_LAYERS)
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_name
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

model.fit(train_images, train_labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(test_images, test_labels),
              shuffle=True,
              callbacks=callbacks)

# Score trained model.
scores = model.evaluate(test_images, test_labels, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])