#import required packages
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense,LSTM, Flatten
import keras
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat

#Plotting confusion matrix
def plot_confusion_matrics(cm, classes, normalize=False, title='Confusion Matrics', cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks=np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=0)
	plt.yticks(tick_marks, classes)
	if normalize:
		cm=cm.astype('float')/cm.sum(axis-1)[:, np.newaxis]
		print('normalized cm')
	else:
		print('without normalization')
	print(cm)
	thresh=cm.max()/2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i,j]>thresh else 'black')
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()
    
#Pre training parameters Configuration
nb_classes = 5
batch_size = 32
epochs = 20

#load features
X=np.load('features/mobilenetv2Train.npy')
X = np.vstack(X)
X=X.reshape(X.shape[0],30,1000)

#Load labels

y = loadmat('features/mobilenetv2trainlabels.mat')
y=y['TrainLabels']

#Split features and labels
X_train,X_val, y_train, y_val = train_test_split(X,y,test_size=0.20)

#Model architecture
input_layer = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm1=LSTM(256, activation='relu', dropout=0.2, return_sequences=True)(input_layer)
lstm2=LSTM(256, activation='relu',dropout=0.2, return_sequences=True)(lstm1)
output_block_1 = keras.layers.add([lstm2, lstm1])
lstm3=LSTM(256, activation='relu',dropout=0.2, return_sequences=True)(output_block_1)
fl=Flatten()(lstm3)
fc1=Dense(128, activation='relu')(fl)
fc1=Dense(128, activation='relu')(fc1)
fc2=Dense(5, activation='softmax')(fc1)
model = keras.models.Model(inputs=input_layer, outputs=fc2)
opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print('Training is started')
print('Please wait this will take a while: ...')
history1=model.fit(X_train,y_train,batch_size = 300,epochs=50, validation_data=(X_val,y_val), verbose=1)
print('Training successfully completed')      

#Model saving 
print('Model saving')    
model.save('Models/mobilenet_residuallstm3.h5')
print('Model saved successfully')  

#Model Testing
print('Start testing the model')
pred=model.predict(X_val)

#Model Evaluation
import numpy as np
accy = history1.history['accuracy']
np_accy = np.array(accy)
score, acc = model.evaluate(X_val, y_val, batch_size=batch_size)
print('Test accuracy:', acc)

#Confusion matrix ploting
y_pred = (pred > 0.5)*1 
confusion_matrix1=confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1))
cm_plot_labels=['Assault', 'Explosion','Fighting','Normal','RoadAccidents']
plot_confusion_matrics(confusion_matrix1, cm_plot_labels, title='Confusion Matrics')

