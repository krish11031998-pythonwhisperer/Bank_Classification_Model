import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os.path
from sklearn.metrics import confusion_matrix
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import GridSearchCV


class Data():

	def __init__(self,filename:str):
		self.X, self.Y = self.data_import(filename)
		print(self.X[0])
		self.catogorical_encoder(self.X, 1)
		self.catogorical_encoder(self.X, 2)
		print(self.X[0])
		self.sc = None
		self.X = self.one_hot_encoder(self.X, 1)
		self.X_train, self.X_test, self.Y_train, self.Y_test = self.train__test_split(self.X, self.Y, 0.75)
		self.X_train, self.X_test = self.feature_scaling(self.X_train, self.X_test)

	def data_import(self, filename: str):
		dataset = pd.read_csv(filename)
		inputs = dataset.iloc[:, 3:13].values
		outputs = dataset.iloc[:, 13].values

		return inputs, outputs

	def catogorical_encoder(self, data, column):
		label_encoder_1 = LabelEncoder()
		data[:, column] = label_encoder_1.fit_transform(data[:, column])

	def one_hot_encoder(self, data, column):
		try:
			onehotencoder = OneHotEncoder(categorical_features=[column])
			data_hotencoded = onehotencoder.fit_transform(data).toarray()
			data_hotencoded = data_hotencoded[:, 1:]
		except FutureWarning:
			print("SOME RANDOM WARNING")
		finally:
			return data_hotencoded

	def train__test_split(self, data_1, data_2, portion):
		data_1_train, data_1_test, data_2_train, data_2_test = train_test_split(data_1, data_2, test_size=(1 - portion),
																				random_state=0)
		return data_1_train, data_1_test, data_2_train, data_2_test

	def feature_scaling(self, x, y):
		x = np.array(x)
		y = np.array(y)
		self.sc = StandardScaler()
		x_scaled = self.sc.fit_transform(x)
		y_scaled = self.sc.transform(y)
		return x_scaled, y_scaled

	def return_data(self):
		return (self.X_train,self.X_test,self.Y_train,self.Y_test)

class ANN():

	def __init__(self,X_tr,X_te,Y_tr,Y_te,epochs:int=100):
		self.epoch = epochs
		self.X_train = X_tr
		self.X_test = X_te
		self.Y_train = Y_tr
		self.Y_test = Y_te
		self.input_dim = len(self.X_train[0])
		self.filename = '/Users/krishnavenkatramani/Desktop/ANN/ANN_tutorial/model.json'
		self.weightsfiledir = '/Users/krishnavenkatramani/Desktop/ANN/ANN_tutorial/model.h5'
		print(self.input_dim)
		if os.path.exists(self.filename) and os.path.exists(self.weightsfiledir):
			self.classifier = self.save_open(action='open')
			self.classifier = self.NN_compile(self.classifier)
		else:
			print("creating a new NN\n")
			self.classifier = self.NN_design()
			self.fit_train()

	def NN_design(self):
		classifier  = tf.keras.Sequential([
			tf.keras.layers.Dense(6,input_shape = (self.input_dim,),kernel_initializer = 'uniform',activation = 'relu'),
			tf.keras.layers.Dropout(0.9),
			tf.keras.layers.Dense(6,kernel_initializer='uniform',activation='relu'),
			tf.keras.layers.Dense(1,kernel_initializer='uniform',activation='sigmoid')
		])

		classifier.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')
		return classifier


	def NN_compile(self,classifier):
		classifier.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')
		return classifier


	def fit_train(self):
		self.classifier.fit(self.X_train,self.Y_train,batch_size=10,epochs = self.epoch,validation_data=(self.X_test,self.Y_test))
		choice = input("Do you want to save the neural network model? \n")
		if choice.lower() == 'y':
			self.save_open(action='save')
		else:
			print("alrighty, continue modelling\n")


	def predict(self,test_data = None):
		if test_data is None:
			Y_predicted  = self.classifier.predict(self.X_test)
			return Y_predicted
		else:
			Y_pred = self.classifier.predict(test_data)
			return Y_pred



	def save_open(self,action:str):
		if action == 'save':
			save_classifier = self.classifier.to_json()
			with open(self.filename,'w') as json_file:
				json_file.write(save_classifier)
			self.classifier.save_weights(self.weightsfiledir)
			print("Successfully saved the NN classifier")
		elif action == 'open':
			with open(self.filename,'r') as json_file:
				json_classifier = json_file.read()
			loaded_classifier = tf.keras.models.model_from_json(json_classifier)
			loaded_classifier.load_weights(self.weightsfiledir)
			print("Sucessfully loaded the Classifier")
			return loaded_classifier

	def accuracy(self):
		Y_pred = self.predict()
		Y_pred = (Y_pred>0.5)
		cm = confusion_matrix(self.Y_test,Y_pred)
		return float((cm[0][0]+cm[1][1])/len(self.Y_test))

	def tuning(self):
		self.HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([3, 6]))
		self.HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
		self.HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

		self.METRIC_ACCURACY = 'accuracy'

		with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
			hp.hparams_config(
				hparams=[self.HP_NUM_UNITS, self.HP_DROPOUT, self.HP_OPTIMIZER],
				metrics=[hp.Metric(self.METRIC_ACCURACY, display_name='Accuracy')],
			)
		session_num = 0

		for num_units in self.HP_NUM_UNITS.domain.values:
			for dropout_rate in (self.HP_DROPOUT.domain.min_value, self.HP_DROPOUT.domain.max_value):
				for optimizer in self.HP_OPTIMIZER.domain.values:
					hparams = {
						self.HP_NUM_UNITS: num_units,
						self.HP_DROPOUT: dropout_rate,
						self.HP_OPTIMIZER: optimizer,
					}
					run_name = "run-%d" % session_num
					print('--- Starting trial: %s' % run_name)
					print({h.name: hparams[h] for h in hparams})
					self.run('logs/hparam_tuning/' + run_name, hparams)
					session_num += 1

	def train_test_model(self,hparams):
		classifier  = tf.keras.Sequential([
			tf.keras.layers.Dense(6,input_shape = (self.input_dim,),kernel_initializer = 'uniform',activation = 'relu'),
			tf.keras.layers.Dropout(hparams[self.HP_DROPOUT]),
			tf.keras.layers.Dense(hparams[self.HP_NUM_UNITS],kernel_initializer='uniform',activation='relu'),
			tf.keras.layers.Dense(1,kernel_initializer='uniform',activation='sigmoid')
		])

		classifier.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')

		classifier.fit(self.X_train, self.Y_train, epochs=1) # Run with 1 epoch to speed things up for demo purposes
		_, accuracy = classifier.evaluate(self.X_test, self.Y_test)
		return accuracy

	def run(self,run_dir, hparams):
		with tf.summary.create_file_writer(run_dir).as_default():
			hp.hparams(hparams)  # record the values used in this trial
			accuracy = self.train_test_model(hparams)
			tf.summary.scalar(self.METRIC_ACCURACY, accuracy, step=1)


if __name__ == "__main__":
	data = Data(filename='/Users/krishnavenkatramani/Desktop/ANN/ANN_tutorial/Churn_Modelling.csv')
	X_tr,X_te,Y_tr,Y_te= data.return_data()
	classifier = ANN(X_tr,X_te,Y_tr,Y_te)
	print("the accuracy of the model is : {}".format(classifier.accuracy()*100))
	new_input = np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])
	new_input = data.sc.transform(new_input)
	Y_pred = classifier.predict(test_data=new_input)
	print(Y_pred)
	Y_pred = (Y_pred>0.5)
	print("The chances of him leaves is : {}".format(Y_pred))
	classifier.tuning()



