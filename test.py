from keras.models import Model
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from preProcess import convnet
import csv




csv_file =  'Data/test.csv'
inputFile = pd.read_csv(csv_file)
sequence = inputFile['sequence']
#label = inputFile['label']
charsPossible = 'ACGT'
charToInt = dict((c, i) for i, c in enumerate(charsPossible))
N = len(sequence)
intEncoded = np.zeros((N,14))
for i in range(len(sequence)):
    intEncoded[i] = [charToInt[char] for char in sequence[i]]
encoded = to_categorical(intEncoded)
oneHotEncoded = np.transpose(encoded,(0,2,1))
oneHotEncodedTest = oneHotEncoded[...,np.newaxis]
print (oneHotEncoded[0])


"""

file = pd.read_csv('Data/test.csv')
sequence = file['sequence']
alphabet = 'ACGT'
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
N = len(sequence)
integer_encoded = np.zeros((N,14))
#print("Starting the one hot encoding process.")
for i in range(len(sequence)):
    integer_encoded[i] = [char_to_int[char] for char in sequence[i]]
encoded = to_categorical(integer_encoded)
onehotenc = np.transpose(encoded,(0,2,1))
onehotenctest = onehotenc[...,np.newaxis]    
"""

print (oneHotEncodedTest.shape)



model =  convnet(input_shape=(4,14,1))
model.load_weights('weight2.h5')
labelsPredicted = model.predict(oneHotEncodedTest,verbose=1,batch_size=128)

print (labelsPredicted[0])

classSeqLabels = np.zeros(labelsPredicted.shape[0])
i=0
for pred in labelsPredicted:
    if pred[0]>=0.5:
        classSeqLabels[i] = 0
    else:
        classSeqLabels[i] = 1
    i+=1
for i,p in enumerate(classSeqLabels):
  classSeqLabels[i] = int(p)
  
  
N = np.shape(classSeqLabels)  
print (classSeqLabels[0])

print("Started writing into the finalSubmission.csv file.")

with open('finalSubmission2.csv', 'w') as myfile:
    #writer = csv.writer(myfile)
    #writer.writerow(["Id", "Prediction"])
    myfile.write("{},{}\n".format("Id", "Prediction"))
    for i in range(400):
        myfile.write("{},{}\n".format(i, int(classSeqLabels[i])))
        
myfile.close()   
print("Finished writing into the finalSubmission.csv file.")   
    
# writer=csv.writer(open('submission.csv','wb'))
# header=['id','prediction']
# length_list=len(header)
# i=0

# while i!=length_list :
    # data=header[i]
    # print data
    # i=i+1
    # writer.writerow(data)
