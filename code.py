import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
	#Load Data
	data = pd.read_csv('IRIS.csv')
	
	#Visualize Data
	x_sepal_l = np.array(data['sepal_length'])
	x_sepal_w = np.array(data['sepal_width'])
	labels = np.array(data['species'])
	
	plt.plot(x_sepal_l[labels == 'Iris-setosa'], x_sepal_w[labels == 'Iris-setosa'], 'ro')
	plt.plot(x_sepal_l[labels == 'Iris-versicolor'], x_sepal_w[labels == 'Iris-versicolor'], 'bo')
	plt.plot(x_sepal_l[labels == 'Iris-virginica'], x_sepal_w[labels == 'Iris-virginica'], 'go')
	
	plt.xlabel('Sepal Length')
	plt.ylabel('Sepal Width')
	plt.title('Iris Dataset')
	plt.show()
	
	#Model Classifier
	clf = SVC(gamma = 'auto')
	
	#Features
	X = np.array(data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
	
	#Labels
	y = np.array(data['species'])
	
	#Split Training and Test Data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
	print('Training data:', X_train.shape)
	print('Test Data:', X_test.shape)
	
	clf.fit(X_train, y_train)
	
	y_pred = clf.predict(X_test)
	print('Model Accuracy:', accuracy_score(y_test, y_pred))
