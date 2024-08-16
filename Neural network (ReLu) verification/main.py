import time
import random 
import Useful_functions as aux
from Verify_MM_fixed_point import verify_matrix_multiplication
from Activations import verify_relu 
import random 
import matplotlib.pyplot as plt 
import pickle 


precision = 8 # power of two 
mantisa = 6  # Power of two minus 2 

items_ours = [2] # Set the prime number and the group order 

           
[prime_number, order, factors, base_group] = aux.prime_order(items_ours[0])         
bias = int((order - 1)/2)  
# numbers are from -(prime_number - 1)/2 to +(prime_number - 1)/2. Thus we need bias in remainder calculations


with open('my_list.pkl', 'rb') as file:  # Model weights and inputs 
    inputs, outputs, W1, W2, W3, W4 = pickle.load(file)



i = 0
[m_time1, p_time1, v_time1, comm1, A] = verify_matrix_multiplication(inputs[i], W1, [prime_number, order, factors, base_group], precision, mantisa)
[m_time2, p_time2, v_time2, comm2, B] = verify_relu(A, [prime_number, order, factors, base_group], precision, mantisa, bias)  
[m_time3, p_time3, v_time3, comm3, C] = verify_matrix_multiplication(B, W2, [prime_number, order, factors, base_group], precision, mantisa)
[m_time4, p_time4, v_time4, comm4, D] = verify_relu(C, [prime_number, order, factors, base_group], precision, mantisa, bias)  
[m_time5, p_time5, v_time5, comm5, E] = verify_matrix_multiplication(D, W3, [prime_number, order, factors, base_group], precision, mantisa)
[m_time6, p_time6, v_time6, comm6, F] = verify_relu(E, [prime_number, order, factors, base_group], precision, mantisa, bias) 
[m_time7, p_time7, v_time7, comm7, G] = verify_matrix_multiplication(F, W4, [prime_number, order, factors, base_group], precision, mantisa)
p_time = p_time1 + p_time2 + p_time3 + p_time4 + p_time5 + p_time6 + p_time7 
v_time = v_time1 + v_time2 + v_time3 + v_time4 + v_time5 + v_time6 + v_time7 
comm = comm1 + comm2 + comm3 + comm4 + comm5 + comm6 + comm7 
m_time = m_time1 + m_time2 + m_time3 + m_time4 + m_time5 + m_time6 + m_time7 
print("Finished!")  
print("m_time = ", m_time)  
print("v_time = ", v_time)  
print("p_time = ", p_time)  
print("comm = ", comm)  







