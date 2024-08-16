# Reading the trained matrices in numpy and save them as lists. We also check the NN with matrix multiplication mode. 

import numpy as np 
from keras.datasets import mnist 
import pickle 
import math 
import time 

def fixed_point(a, mantisa, precision): 
    result = round(a * 2**precision)  
    if abs(result) >= 2**(mantisa + precision): 
        print("Warning!")
        return False 
    else:
        return result  


def fixed_point_matrix(a, mantisa, precision): 
    b = []
    for i in range(a.shape[0]):
        c = []
        for j in range(a.shape[1]):
            c.append(fixed_point(a[i, j], mantisa, precision))
        b.append(c)
    return b

def relu(x):
    a = np.zeros((len(x), len(x[0]))) 
    for i in range(len(x)):
        for j in range(len(x[0])):
            a[i, j] = max(0, x[i][j])
    return a 

def relu_fixed_point(x):
    a = np.zeros((len(x), len(x[0])), "int") 
    for i in range(len(x)):
        for j in range(len(x[0])):
            a[i, j] = max(0, x[i][j])
    return a 

def error_vector(a, precision): 
    b = []
    if precision < 1:
        for i in range(len(a)):
            c = []
            for j in range(len(a[0])):
                c.append(0) 
            b.append(c)    
    else: 
        d = 2**(precision-1) 
        e = 2 * d 
        for i in range(len(a)): 
            c = [] 
            for j in range(len(a[0])): 
                c.append(((a[i][j] + d) % e) - d) 
            b.append(c) 
    return b 

def round_vector(a, e, precision): 
    c = []    
    b = [] 
    for i in range(len(a)):
        b = [] 
        for j in range(len(a[0])):
            b.append((a[i][j] - e[i][j]) >> precision)   
        c.append(b) 
    return c 


def matmul_list(a, b):  
    c = []  
    if len(a[0]) != len(b):  
        print("Error: Dimension mismatch!")  
    for i in range(len(a)):  
        r = []  
        for j in range(len(b[0])):  
            d = 0  
            for k in range(len(a[0])):  
                d = d + a[i][k] * b[k][j] 
            r.append(d)  
        c.append(r)  
    return c  
 

def fixed_point_MatMul(input_renov, A_renov, precision): 
    B_renov = matmul_list(input_renov, A_renov) 
    V_error_1 = error_vector(B_renov, precision) 
    C_renov = round_vector(B_renov, V_error_1, precision) 
    return C_renov 


precision = 8 
mantisa = 6


# Load MNIST handwritten digit data
(X_train, y_train), (X_test, y_test) = mnist.load_data() 


X_train = X_train.astype('float32') / 255  
X_test = X_test.astype('float32') / 255 


# Load W1 to W4. These are generated from third.py
W1_copy = np.load("./matrices/W1.npy") 
W1 = np.zeros((1024, W1_copy.shape[1])) 
W1[0:W1_copy.shape[0],:] = W1_copy 
W2 = np.load("./matrices/W2.npy") 
W3 = np.load("./matrices/W3.npy") 
W4_copy = np.load("./matrices/W4.npy") 
W4 = np.zeros((W4_copy.shape[0], 16))
W4[:,0:W4_copy.shape[1]] = W4_copy 

# To fixed point arithmetic
W1_fixed_point = fixed_point_matrix(W1, precision=precision, mantisa=mantisa) 
W2_fixed_point = fixed_point_matrix(W2, precision=precision, mantisa=mantisa) 
W3_fixed_point = fixed_point_matrix(W3, precision=precision, mantisa=mantisa) 
W4_fixed_point = fixed_point_matrix(W4, precision=precision, mantisa=mantisa) 


counter1 = 0 
counter2 = 0 
counter3 = 0 
counter4 = 0 
counter5 = 0 
# Run the NN model in exact form 
for i in range(100): 
    img = np.zeros((1, 1024))
    img[0, 0:784] = X_train[i].reshape(1, 784) 
    L1 = relu(np.matmul(img, W1)) 
    L2 = relu(np.matmul(L1, W2)) 
    L3 = relu(np.matmul(L2, W3)) 
    L4 = np.matmul(L3, W4) 


    t1 = time.time()
    img_fixed_point = fixed_point_matrix(img, precision=precision, mantisa=mantisa) 
    # Run the NN model in fixed-point form 
    L1_fixed_point = relu_fixed_point(fixed_point_MatMul(img_fixed_point, W1_fixed_point, precision)) 
    L2_fixed_point = relu_fixed_point(fixed_point_MatMul(L1_fixed_point, W2_fixed_point, precision)) 
    L3_fixed_point = relu_fixed_point(fixed_point_MatMul(L2_fixed_point, W3_fixed_point, precision)) 
    L4_fixed_point = fixed_point_MatMul(L3_fixed_point, W4_fixed_point, precision) 
    t2 = time.time() 
    print("Running time (ms) = ", 1000 * (t2 - t1))

    if y_train[i] == np.argmax(L4) == np.argmax(L4_fixed_point[0][0:10]): 
        counter1 = counter1 + 1 
    elif y_train[i] == np.argmax(L4): 
        counter2 = counter2 + 1 
        print("i = ", i)
    elif np.argmax(L4) == np.argmax(L4_fixed_point[0][0:10]): 
        counter3 = counter3 + 1 
    elif np.argmax(L4) == np.argmax(L4_fixed_point[0][0:10]):
        counter4 = counter4 + 1
    else:
        counter5 = counter5 + 1
        

print("Good result = ", counter1) 
print("Fixed point caused error = ", counter2) 
print("Model training isn't sufficient = ", counter3) 
print("Wierd! fixed point made better result = ", counter4) 
print("Scattered result = ", counter5) 




input_data = []
output_data = []
for i in range(100): 
    img = np.zeros((1, 1024))
    img[0, 0:784] = X_train[i].reshape(1, 784)     
    img_fixed_point = fixed_point_matrix(img, precision=precision, mantisa=mantisa) 
    input_data.append(img_fixed_point)
    output_data.append(y_train[i])


# Save fixed-point version 
with open('my_list.pkl', 'wb') as file:
    pickle.dump((input_data, output_data, W1_fixed_point, W2_fixed_point, W3_fixed_point, W4_fixed_point), file)

with open('my_list.pkl', 'rb') as file:
    inputs, outputs, W1, W2, W3, W4 = pickle.load(file)




