import numpy as np
import prover as p
import auxiliary_functions as aux

# -------------------------------------------- Initialization
prime_number = 2^64 - 2^32 + 1 
n = 8  # Vector length 
precision = 5 # The number of bits 
f = 0 
Vector_error = (np.random.rand(n) * 2**(precision)).astype(int) 
Sign_vector = np.sign((np.random.rand(n) * 2 - 1))
Vector_error_signed = Vector_error * Sign_vector
Binary_error = ((Vector_error.reshape(-1,1) & (2**np.arange(precision - 1, -1, -1))) != 0).astype(int)
Sign_vector = np.tile(Sign_vector.reshape(-1,1), (1, precision))
Binary_error = Sign_vector * Binary_error 
Binary_error = Binary_error.astype(int)

# --------------------------------------------
# Prover:
E = np.zeros(tuple([precision] + [2 for i in range(np.log2(Binary_error.shape[0]).astype(int))]))

for j in range(precision):
    E[j] = aux.binary_reshape_vector(Binary_error[:,j]) 
I = aux.binary_reshape(np.identity(n))

# -------------------------------------------- Start sumcheck  
# Verifier:
x = prime_number # range of numbers 
l = int(np.log2(n))
r_1 = np.ceil(np.random.rand(l) * x) # First random input chosen by the verifier at the begining of sumcheck
a = np.ceil(np.random.rand(l) * x) # Second random input chosen by the verifier step by step in each round of sumcheck 
S = np.zeros((precision))

first_part = p.MLE(I, np.concatenate((r_1, a), axis=0))
for j in range(precision):
    second_part = p.MLE(E[j], a)
    S[j] = np.remainder(first_part *  second_part * (1 - second_part) * (1 + second_part), prime_number) # Final answers 


# --------------------------------------------
# Prover:
for i in range(l): 
    I = p.squeeze_table_l(I, r_1[i])


gj_0 = np.zeros(precision)
gj_1 = np.zeros(precision)
gj_2 = np.zeros(precision)
gj_3 = np.zeros(precision)
gj_4 = np.zeros(precision)
g0 = np.zeros(precision)
g1 = np.zeros(precision)
g2 = np.zeros(precision)
g3 = np.zeros(precision)
g4 = np.zeros(precision)

for j in range(precision):
    e = E[j]
    gj_0[j] = p.sum_table(np.multiply(I[0,:], e[0,:] * (1 - e[0,:]) * (1 + e[0,:]))) 
    gj_1[j] = p.sum_table(np.multiply(I[1,:], e[1,:] * (1 - e[1,:]) * (1 + e[1,:])))
    second_part = np.remainder(2 * e[1,:] - e[0,:], prime_number) 
    gj_2[j] = p.sum_table(np.multiply(2 * I[1,:] - I[0,:], second_part * (1 - second_part) * (1 + second_part)))
    second_part = np.remainder(3 * e[1,:] - 2 * e[0,:], prime_number) 
    gj_3[j] = p.sum_table(np.multiply(3 * I[1,:] - 2 * I[0,:], second_part * (1 - second_part) * (1 + second_part)))
    second_part = np.remainder(4 * e[1,:] - 3 * e[0,:], prime_number)
    gj_4[j] = p.sum_table(np.multiply(4 * I[1,:] - 3 * I[0,:], second_part * (1 - second_part) * (1 + second_part)))


# --------------------------------------------
# Verifier:
for j in range(precision):
    if np.remainder(gj_0[j] + gj_1[j], prime_number) != 0:
        print("gj_0[j] = ", gj_0[j])
        print("gj_1[j] = ", gj_1[j])
        print("Error 1!")
        f = 1

g0[:] = gj_0[:]
g1[:] = gj_1[:]
g2[:] = gj_2[:]
g3[:] = gj_3[:]
g4[:] = gj_4[:]


for i in range(l - 2):
    # Prover:
    I = p.squeeze_table_l(I, a[i])
    E_new = np.zeros(tuple([precision] + [2 for k in range(l - 1 - i)])) 
    for j in range(precision): 
        E_new[j] = p.squeeze_table_l(E[j], a[i])
        e = E_new[j]
        gj_0[j] = p.sum_table(np.multiply(I[0,:], e[0,:] * (1 - e[0,:]) * (1 + e[0,:])))
        gj_1[j] = p.sum_table(np.multiply(I[1,:], e[1,:] * (1 - e[1,:]) * (1 + e[1,:])))
        second_part = np.remainder(2 * e[1,:] - e[0,:], prime_number) 
        gj_2[j] = p.sum_table(np.multiply(2 * I[1,:] - I[0,:], second_part * (1 - second_part) * (1 + second_part)))
        second_part = np.remainder(3 * e[1,:] - 2 * e[0,:], prime_number) 
        gj_3[j] = p.sum_table(np.multiply(3 * I[1,:] - 2 * I[0,:], second_part * (1 - second_part) * (1 + second_part)))   
        second_part = np.remainder(4 * e[1,:] - 3 * e[0,:], prime_number)
        gj_4[j] = p.sum_table(np.multiply(4 * I[1,:] - 3 * I[0,:], second_part * (1 - second_part) * (1 + second_part))) 
    

    E = E_new
    # --------------------------------------------
    # Verifier: 
    for j in range(precision):
        if np.remainder(24 * (gj_0[j] + gj_1[j]), prime_number) != p.quinary_single_var_eval(g0[j], g1[j], g2[j], g3[j], g4[j], a[i]):
            print("Error 2!")
            f = 1
    g0[:] = gj_0[:]
    g1[:] = gj_1[:]
    g2[:] = gj_2[:]
    g3[:] = gj_3[:]
    g4[:] = gj_4[:]


# --------------------------------------------
# Prover:
I = p.squeeze_table_l(I, a[l - 2])
E_new = np.zeros(tuple([precision] + [2 for k in range(1)])) 
for j in range(precision): 
    E_new[j] = p.squeeze_table_l(E[j], a[l-2])
    e = E_new[j]
    gj_0[j] = np.remainder(np.multiply(I[0], e[0] * (1 - e[0]) * (1 + e[0])), prime_number)
    gj_1[j] = np.remainder(np.multiply(I[1], e[1] * (1 - e[1]) * (1 + e[1])), prime_number)
    second_part = np.remainder(2 * e[1] - e[0], prime_number) 
    gj_2[j] = np.remainder(np.multiply(2 * I[1] - I[0], second_part * (1 - second_part)  * (1 + second_part)), prime_number)
    second_part = np.remainder(3 * e[1] - 2 * e[0], prime_number) 
    gj_3[j] = np.remainder(np.multiply(3 * I[1] - 2 * I[0], second_part * (1 - second_part)  * (1 + second_part)), prime_number)
    second_part = np.remainder(4 * e[1] - 3 * e[0], prime_number)
    gj_4[j] = np.remainder(np.multiply(4 * I[1] - 3 * I[0], second_part * (1 - second_part) * (1 + second_part)), prime_number) 
E = E_new

# --------------------------------------------
# Verifier: 
for j in range(precision):
    if np.remainder(24 * (gj_0[j] + gj_1[j]), prime_number) != p.quinary_single_var_eval(g0[j], g1[j], g2[j], g3[j], g4[j], a[l - 2]):
        print("Error 3!")
        f = 1
g0[:] = gj_0[:]
g1[:] = gj_1[:]
g2[:] = gj_2[:]
g3[:] = gj_3[:]
g4[:] = gj_4[:]
# --------------------------------------------
# Verifier: 
for j in range(precision):
    s = S[j]
    if np.remainder(24 * s, prime_number) != p.quinary_single_var_eval(g0[j], g1[j], g2[j], g3[j], g4[j], a[l - 1]):
        print("Error 4!")
        f = 1

if f == 0:
    print("Done!")    
  
      
