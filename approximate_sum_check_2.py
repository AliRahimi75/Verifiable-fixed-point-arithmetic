import numpy as np
import prover as p
import auxiliary_functions as aux

# -------------------------------------------- Initialization
n = 16
precision = 5
f = 0
Vector_error = (np.random.rand(n) * 2**(precision + 1)).astype(int)

Binary_error = ((Vector_error.reshape(-1,1) & (2**np.arange(precision))) != 0).astype(int)

# --------------------------------------------
# Prover:
E = np.zeros(tuple([precision] + [2 for i in range(np.log2(Binary_error.shape[0]).astype(int))]))

for j in range(precision):
    E[j] = aux.binary_reshape_vector(Binary_error[:,j]) 
I = aux.binary_reshape(np.identity(n))

# -------------------------------------------- Start sumcheck
# Verifier:
x = 10
r_1 = np.ceil(np.random.rand(int(np.log2(n))) * 10)
l = int(np.log2(n))
a = np.ceil(np.random.rand(int(np.log2(n))) * 10)
S = np.zeros((precision))

first_part = p.MLE(I, np.concatenate((r_1, a), axis=0))
for j in range(precision):
    second_part = p.MLE(E[j], a)
    S[j] = first_part *  second_part * (1 - second_part) # Final answers


# --------------------------------------------
# Prover:
for i in range(int(np.log2(n))): 
    I = p.squeeze_table_l(I, r_1[i])


primary_assertion = 0

gj_0 = np.zeros(precision)
gj_1 = np.zeros(precision)
gj_2 = np.zeros(precision)
gj_3 = np.zeros(precision)
g0 = np.zeros(precision)
g1 = np.zeros(precision)
g2 = np.zeros(precision)
g3 = np.zeros(precision)

for j in range(precision):
    e = E[j]
    gj_0[j] = p.sum_table(np.multiply(I[0,:], e[0,:] * (1 - e[0,:]))) 
    gj_1[j] = p.sum_table(np.multiply(I[1,:], e[1,:] * (1 - e[1,:])))
    second_part = 2 * e[1,:] - e[0,:] 
    gj_2[j] = p.sum_table(np.multiply(2 * I[1,:] - I[0,:], second_part * (1 - second_part)))
    second_part = 3 * e[1,:] - 2 * e[0,:] 
    gj_3[j] = p.sum_table(np.multiply(3 * I[1,:] - 2 * I[0,:], second_part * (1 - second_part)))


# --------------------------------------------
# Verifier:
for j in range(precision):
    if gj_0[j] + gj_1[j] != 0:
        print("Error 1!")
        f = 1

g0[:] = gj_0[:]
g1[:] = gj_1[:]
g2[:] = gj_2[:]
g3[:] = gj_3[:]


for i in range(l - 2):
    # Prover:
    I = p.squeeze_table_l(I, a[i])
    E_new = np.zeros(tuple([precision] + [2 for k in range(l - 1 - i)])) 
    for j in range(precision): 
        E_new[j] = p.squeeze_table_l(E[j], a[i])
        e = E_new[j]
        gj_0[j] = p.sum_table(np.multiply(I[0,:], e[0,:] * (1 - e[0,:])))
        gj_1[j] = p.sum_table(np.multiply(I[1,:], e[1,:] * (1 - e[1,:])))
        second_part = 2 * e[1,:] - e[0,:] 
        gj_2[j] = p.sum_table(np.multiply(2 * I[1,:] - I[0,:], second_part * (1 - second_part)))
        second_part = 3 * e[1,:] - 2 * e[0,:] 
        gj_3[j] = p.sum_table(np.multiply(3 * I[1,:] - 2 * I[0,:], second_part * (1 - second_part)))   
    

    E = E_new
    # --------------------------------------------
    # Verifier: 
    for j in range(precision):
        if 6 * (gj_0[j] + gj_1[j]) != p.senary_single_var_eval(g0[j], g1[j], g2[j], g3[j], a[i]):
            print("Error 2!")
            f = 1
    g0[:] = gj_0[:]
    g1[:] = gj_1[:]
    g2[:] = gj_2[:]
    g3[:] = gj_3[:]


# --------------------------------------------
# Prover:
I = p.squeeze_table_l(I, a[l - 2])
E_new = np.zeros(tuple([precision] + [2 for k in range(1)])) 
for j in range(precision): 
    E_new[j] = p.squeeze_table_l(E[j], a[l-2])
    e = E_new[j]
    gj_0[j] = np.multiply(I[0], e[0] * (1 - e[0]))
    gj_1[j] = np.multiply(I[1], e[1] * (1 - e[1]))
    second_part = 2 * e[1] - e[0] 
    gj_2[j] = np.multiply(2 * I[1] - I[0], second_part * (1 - second_part))
    second_part = 3 * e[1] - 2 * e[0] 
    gj_3[j] = np.multiply(3 * I[1] - 2 * I[0], second_part * (1 - second_part))
E = E_new

# --------------------------------------------
# Verifier: 
for j in range(precision):
    if 6 * (gj_0[j] + gj_1[j]) != p.senary_single_var_eval(g0[j], g1[j], g2[j], g3[j], a[l - 2]):
        print("Error 3!")
        f = 1
g0[:] = gj_0[:]
g1[:] = gj_1[:]
g2[:] = gj_2[:]
g3[:] = gj_3[:]
# --------------------------------------------
# Verifier: 
for j in range(precision):
    s = S[j]
    if p.senary_single_var_eval(g0[j], g1[j], g2[j], g3[j], a[l - 1]) != 6 * s:
        print("Error 4!")
        f = 1

if f == 0:
    print("Done!")    
  
      