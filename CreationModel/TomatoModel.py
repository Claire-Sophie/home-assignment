from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import ResNet101
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dense,  Dropout
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np


train_dir = 'Training'
validation_dir = 'Test'

def get_labe(train_folder):
    label = {}
    for i in range(len(os.listdir(train_folder))):
        label[i] = os.listdir(train_folder)[i]
    return label

def process_data(train_data , validation_data , batch_size , target_size ):

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

    output = Dense(5, activation='softmax')(model_tomato)
    model_final = Model(inputs=model_pre.input, outputs=output)

    return model_final, model_pre

def has_tomato(path):
    lab = get_labe(train_dir)
    img = image.load_img(path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    prediction = model_final.predict(img_tensor)

    maxindex = int(np.argmax(prediction))

    label_pred = None
    for i, j in lab.items():
        if i == maxindex:
            label_pred = j

    return label_pred


if __name__ == '__main__':


    train , val = process_data(train_dir , validation_dir , 32 , (224 ,224))
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

    path = 'AL001-02 tomate.jpg'

    print(has_tomato(path))

