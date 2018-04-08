import numpy as np
import pandas as pd
from keras.utils import to_categorical

csv_file =  'Data/train.csv'
inputFile = pd.read_csv(csv_file)
sequence = inputFile['sequence']
label = inputFile['label']
charsPossible = 'ACGT'
charToInt = dict((c, i) for i, c in enumerate(charsPossible))
N = len(sequence)
intEncoded = np.zeros((N,14))
for i in range(len(sequence)):
    intEncoded[i] = [charToInt[char] for char in sequence[i]]
encoded = to_categorical(intEncoded)
oneHotEncoded = np.transpose(encoded,(0,2,1))
oneHotEncodedTrain = oneHotEncoded[...,np.newaxis]
print (oneHotEncoded[0])

labelVector = to_categorical(label)
#print (labelVector[1])



    
    


from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Reshape, Flatten
from keras.layers import Lambda, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import losses,optimizers,utils
from keras.optimizers import Adam




def convnet(input_shape):
    dropout = 0.5
    inputUsed = Input(shape = input_shape, name = 'input1')
    x = Conv2D(32,(3,3),padding = 'same')(inputUsed)
    x = BatchNormalization()(x) 
    x = Activation('relu')(x)
    #x = Dropout(dropout)(x)
    print (K.shape(x))
    x = Conv2D(64,(3,3),padding = 'same')(x)
    x = BatchNormalization()(x) 
    x = Activation('relu')(x)
    #x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)
    print (K.shape(x))
    x = Flatten()(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(2,activation='softmax')(x)
    model = Model(inputUsed, x)
    return model
    
model = convnet(input_shape=(4,14,1))
model.summary()
model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint('weight2.h5', monitor='val_loss',save_best_only=True)
model.save('my_model2.h5')
    
model.fit(oneHotEncodedTrain,labelVector,batch_size=128, epochs=500, verbose=1, shuffle=True,validation_split=0.25,callbacks=[checkpoint])