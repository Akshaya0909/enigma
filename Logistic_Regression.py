import pandas as pd 
import numpy as np 
data=pd.read_csv('User_Data.csv',sep=',') #Uploaded the dataset in colab. Please upload it again if you are running the code again
data['split'] = np.random.randn(data.shape[0],1)
msk = np.random.rand(len(data)) <=0.7
data_train = data[msk]
data_test = data[~msk]
X_train=data_train.iloc[:,[2,3]].values
Y_train = data_train.iloc[:,4].values
X_test=data_test.iloc[:,[2,3]].values
Y_test = data_test.iloc[:,4].values
X_train = (X_train-X_train.mean())/X_train.std() 
X_test=(X_test-X_test.mean())/X_test.std()      
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
Y_train = Y_train[:, np.newaxis]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
Y_test = Y_test[:, np.newaxis]
theta = np.ones((X_train.shape[1], 1)) 
learning_rate=0.4

def sigmoid(x):  # sigmoid function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def weighted_input(theta, x): # Computes the weighted sum of inputs
    return np.dot(x, theta)


def probability(theta, x): # We assume the output of sigmoid to be posterior probability
    return sigmoid(weighted_input(theta, x))

def cost_function(theta, x, y):
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost

def gradient(theta, x, y):
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(weighted_input(theta,   x)) - y) 

def hessian(x,y,theta):
    xTrans = x.transpose()                                      
    sig = sigmoid(np.dot(x,theta))                              
    result = (1.0/len(x) * np.dot(xTrans, x) * np.diag(sig) * np.diag(1 - sig) )   
    return result

def updateTheta(x,y,theta):
    hessianInv = np.linalg.pinv(hessian(x,y,theta))                         
    grad = gradient(theta,x,y)                                  
    theta = theta - np.matmul(hessianInv, grad)                     
    return theta

def grad_descent(x,y,theta):
    grad = gradient(theta,x,y)
    theta = theta - learning_rate*grad                   
    return theta

weighted_input(theta,X_train)
probability(theta,X_train)
for i in range(100):
    costResult = cost_function(theta,X_train,Y_train)
    grad = gradient(theta,X_train,Y_train)
    theta = grad_descent(X_train,Y_train,theta)
    cost_train=cost_function(theta,X_train,Y_train)

print('weights computed using gradient descent :\n',theta) 
print('Prediction Cost computed using gradient descent on training data :',cost_train)

for i in range(100):
    costResult = cost_function(theta,X_train,Y_train)
    grad = gradient(theta,X_train,Y_train)
    theta = grad_descent(X_train,Y_train,theta)
    cost_test=cost_function(theta,X_train,Y_train)

print('weights computed using gradient descent :\n',theta) 
print('Prediction Cost computed using gradient descent on test data :',cost_test)

for i in range(2):
    costResult = cost_function(theta,X_test,Y_test)
    hessianResult = hessian(X_test,Y_test,theta)
    grad = gradient(theta,X_train,Y_train)
    theta = updateTheta(X_test,Y_test,theta)
    costResult_train=cost_function(theta,X_test,Y_test)
print('weights computed using hessian matrix  :\n',theta)  
print('Prediction Cost computed using hessian matrix on train data :',costResult_train)
for i in range(1):
    costResult = cost_function(theta,X_test,Y_test)
    hessianResult = hessian(X_test,Y_test,theta)
    grad = gradient(theta,X_train,Y_train)
    theta = updateTheta(X_test,Y_test,theta)
    costResult_test=cost_function(theta,X_test,Y_test)
print('weights computed using hessian matrix are :\n',theta)   
print('Prediction Cost computed using hessain matrix on test data :',costResult_test)
