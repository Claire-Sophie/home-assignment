
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50 , ResNet101
from keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,  Dropout
from tensorflow.keras.optimizers import Adam

train_dir = 'Training'
validation_dir = 'Test'

def get_labe():
    None

def process_data(train_data , validation_data , batch_size , target_size ):

    #datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_datagen = ImageDataGenerator( rescale=1. / 255., rotation_range=40, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                        horizontal_flip=True, fill_mode='nearest')

    test_datagen = ImageDataGenerator( rescale=1.0 / 255.)

    train_gen = train_datagen.flow_from_directory(train_data, batch_size=batch_size, class_mode='categorical',
                                                        target_size=target_size)
    validation_gen = test_datagen.flow_from_directory(validation_data, batch_size=batch_size,
                                                            class_mode='categorical', target_size=target_size)
    return train_gen , validation_gen


def create_model():
    model_pre =  ResNet101(input_shape=(224 , 224 ,3) , include_top=False, weights='imagenet' , pooling='max')
    model_tomato = model_pre.output

    model_tomato = Dense(128, activation='relu')(model_tomato)
    model_tomato = Dropout(0.5)(model_tomato)
    model_tomato = Dense(128, activation='relu')(model_tomato)
    model_tomato = Dropout(0.5)(model_tomato)

    output = Dense(6, activation='softmax')(model_tomato)
    model_final = Model(inputs=model_pre.input, outputs=output)

    return model_final, model_pre

if __name__ == '__main__':


    train , val = preprocess_input(train_dir , validation_dir , 32 , (224 ,224))
    model_final , model_pre = create_model()

    for layer in model_pre.layers:
        layer.trainable = False

    model_final.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model_final.compile(Adam(lr=0.0001, decay=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])

    EarlyStop = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')

    CheckPoint = ModelCheckpoint(filepath='model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    train_history = model_final.fit \
        (train, steps_per_epoch=len(train), epochs=20,
         validation_data=val, validation_steps=len(val),
         verbose=2, callbacks=[EarlyStop, CheckPoint])



