import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import load_model
import tensorflow as tf

# Good address: https://valueml.com/get-the-output-of-each-layer-in-keras/ 
# Good address: https://github.com/csbanon/mnist-classifiers/blob/main/mnist-digits/mnist-digit-classification-with-a-fully-connected-neural-network.ipynb 


def sig(x): 
    return 1/(1 + np.exp(-x)) 

def soft(x):
    return np.exp(x)/sum(np.exp(x))

def quad(x):
    return x**2

def relu(x):
    a = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a[i, j] = max(0, x[i, j])
    return a 

# Load MNIST handwritten digit data
(X_train, y_train), (X_test, y_test) = mnist.load_data() 

X_train = X_train.astype('float32') / 255 
X_test = X_test.astype('float32') / 255

# Convert y_train into one-hot format
temp = []
for i in range(len(y_train)):
    temp.append(to_categorical(y_train[i], num_classes=10))
y_train = np.array(temp)
# Convert y_test into one-hot format
temp = []
for i in range(len(y_test)):    
    temp.append(to_categorical(y_test[i], num_classes=10))
y_test2 = np.array(temp)


# Create simple Neural Network model
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(8, activation='relu', use_bias=False)) 
model.add(Dense(64, activation='relu', use_bias=False)) 
model.add(Dense(32, activation='relu', use_bias=False)) 
# model.add(Dense(64, activation=lambda x: x**2, use_bias=False)) 
model.add(Dense(10, activation='softmax', use_bias=False)) 
# model.add(Dense(10, activation=lambda x: x**2, use_bias=False)) 


model.summary()

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])
# model.compile()


model.fit(X_train, y_train, epochs=5, batch_size=128,
          validation_data=(X_test,y_test2))

print("\n \n \n \n ")
# model.save("my_model.keras")


"""
predictions = model.predict(X_test)

predictions = np.argmax(predictions, axis=1)
print(predictions)

fig, axes = plt.subplots(ncols=3, sharex=False,
                         sharey=True, figsize=(20, 4))
for i in range(3):
    axes[i].set_title("pred:" + str(predictions[i]) + "\n label:" + str(y_test[i]))
    axes[i].imshow(X_test[i], cmap='gray')
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)
plt.show()
"""


# model = load_model("my_model.keras", safe_mode=False)



layer_output=model.get_layer('flatten').output  #get the Output of the Layer
intermediate_model=tf.keras.models.Model(inputs=model.input, outputs=layer_output) #Intermediate model between Input Layer and Output Layer which we are concerned about
intermediate_prediction=intermediate_model.predict(X_train[4].reshape(1,28,28,1)) #predicting in the Intermediate Node
matrix_way_generated = X_train[4].reshape(1,784)
print("NN: \n", intermediate_prediction, "\n Layer 0 = \n", matrix_way_generated) 

a = model.layers[1].get_weights() 
print("a[0].shape = ", a[0].shape) 
np.save("./matrices/W1", a[0]) 
layer_output=model.get_layer('dense').output  #get the Output of the Layer
intermediate_model=tf.keras.models.Model(inputs=model.input, outputs=layer_output) #Intermediate model between Input Layer and Output Layer which we are concerned about 
intermediate_prediction=intermediate_model.predict(X_train[4].reshape(1,28,28,1)) #predicting in the Intermediate Node 
matrix_way_generated = relu(np.matmul(X_train[4].reshape(1,784), a[0])) 
print("NN: \n", intermediate_prediction, "\n Layer 1 = \n", matrix_way_generated) 
L1 = matrix_way_generated.copy() 

b = model.layers[2].get_weights() 
print("b[0].shape = ", b[0].shape) 
np.save("./matrices/W2", b[0]) 

layer_output=model.get_layer('dense_1').output  #get the Output of the Layer
intermediate_model=tf.keras.models.Model(inputs=model.input, outputs=layer_output) #Intermediate model between Input Layer and Output Layer which we are concerned about
intermediate_prediction=intermediate_model.predict(X_train[4].reshape(1,28,28,1)) #predicting in the Intermediate Node
matrix_way_generated = relu(np.matmul(L1, b[0]))
print("NN: \n", intermediate_prediction, "\n Layer 2 = \n", matrix_way_generated)
L2 = matrix_way_generated.copy()

c = model.layers[3].get_weights() 
print("c[0].shape = ", c[0].shape) 
np.save("./matrices/W3", c[0]) 

layer_output=model.get_layer('dense_2').output  #get the Output of the Layer
intermediate_model=tf.keras.models.Model(inputs=model.input, outputs=layer_output) #Intermediate model between Input Layer and Output Layer which we are concerned about
intermediate_prediction=intermediate_model.predict(X_train[4].reshape(1,28,28,1)) #predicting in the Intermediate Node
matrix_way_generated = relu(np.matmul(L2, c[0]))
print("NN: \n", intermediate_prediction, "\n Layer 3 = \n", matrix_way_generated)
L3 = matrix_way_generated.copy() 


d = model.layers[4].get_weights() 
print("d[0].shape = ", d[0].shape) 
np.save("./matrices/W4", d[0]) 

layer_output=model.get_layer('dense_3').output  #get the Output of the Layer
intermediate_model=tf.keras.models.Model(inputs=model.input, outputs=layer_output) #Intermediate model between Input Layer and Output Layer which we are concerned about
intermediate_prediction=intermediate_model.predict(X_train[4].reshape(1,28,28,1)) #predicting in the Intermediate Node
matrix_way_generated = soft(np.matmul(L3, d[0])[0])
print("NN: \n", intermediate_prediction, "\n Layer 4 = \n", matrix_way_generated)
L4 = matrix_way_generated.copy() 


print("X_train[4].shape = ", X_train[4].shape)
print("NN = \n", np.argmax(model.predict(X_train[4].reshape(1,28,28,1)), axis=1), "\n Output = \n", np.argmax(L4))


# input = X_train[4].reshape(1,784)
# W1 = model.layers[1].get_weights()[0] 
# L1 = sig(np.matmul(input, W1)) OR Activation(np.matmul(input, W1)) 
# W2 = model.layers[2].get_weights()[0]
# Output = np.argmax(soft(np.matmul(L1, W2)[0])) 



