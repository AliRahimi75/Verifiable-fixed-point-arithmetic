import numpy as np
import prover as p
import time
import auxiliary_functions as aux

def Thaler_method(A, B):
    prover_time = 0
    prover_main_load = 0
    verifier_time = 0
    # --------------------------------------------
    # Setup: given two matrices A and B
    n = A.shape[0] # must be a power of two and at least 8
    m = A.shape[1] # must be a power of two and at least 8
    o = B.shape[1] # must be a power of two and at least 8
    if n < 8 or m < 8 or o < 8:
        print("Matrices are not big enough!")
    if np.log2(n) - int(np.log2(n)) != 0:
        print("Sumcheck Error! The first dimension of A is not a power of 2")
    if np.log2(m) - int(np.log2(m)) != 0:
        print("Sumcheck Error! The second dimension of A is not a power of 2") 
    if np.log2(o) - int(np.log2(o)) != 0:
        print("Sumcheck Error! The first dimension of B is not a power of 2")        
    if A.shape[1] != B.shape[0]:
        print("Sumcheck Error! A and B cannot be multipied due to dimension mismatch!")                      
    f = 0 # Verifier error flag
    x = 10 # Precision
    # --------------------------------------------
    # Prover: I've computed A.B correctly.
    t_old = time.time()
    C = aux.binary_reshape(np.matmul(A,B))
    A = aux.binary_reshape(A)
    B = aux.binary_reshape(B)
    t_new = time.time()
    prover_time = prover_time + t_new - t_old
    prover_main_load = prover_main_load + t_new - t_old
    # --------------------------------------------
    # Verifier: Here you are: r_1 and r_2. Give me \tilde{f}_C(r_1,r_2)
    t_old = time.time()
    r_1 = np.random.rand(int(np.log2(n)))
    r_1 = np.ceil(r_1 * x)
    r_2 = np.random.rand(int(np.log2(o)))
    r_2 = np.ceil(r_2 * x)
    # -------------------------------------------- Start sumcheck
    # Verifier: 
    l = int(np.log2(m))
    a = np.random.rand(l)
    a = np.ceil(a * x)
    s = p.MLE(A, np.concatenate((r_1, a), axis=0)) * p.MLE(B, np.concatenate((a, r_2), axis=0))
    t_new = time.time()
    verifier_time = verifier_time + t_new - t_old
    # --------------------------------------------
    # Prover:
    t_old = time.time() 
    for i in range(int(np.log2(n))):
        A = p.squeeze_table_l(A, r_1[i])
    for i in reversed(range(int(np.log2(o)))):
        B = p.squeeze_table_r(B, r_2[i])    

    C1 = p.MLE(C, np.concatenate((r_1,r_2), axis=0))
    gj_0 = p.sum_table(np.multiply(A[0,:], B[0,:]))
    gj_1 = p.sum_table(np.multiply(A[1,:], B[1,:]))
    gj_2 = p.sum_table(np.multiply(2 * A[1,:] - A[0,:], 2 * B[1,:] - B[0,:]))
    t_new = time.time()
    prover_time = prover_time + t_new - t_old
    # --------------------------------------------
    # Verifier:
    t_old = time.time()
    if gj_0 + gj_1 != C1:
        print("Error 1!")
        f = 1
    g0 = gj_0
    g1 = gj_1
    g2 = gj_2
    t_new = time.time()
    verifier_time = verifier_time + t_new - t_old    
    # --------------------------------------------
    for i in range(l - 2):
        # Prover:
        t_old = time.time()
        A = p.squeeze_table_l(A, a[i])
        B = p.squeeze_table_l(B, a[i])
        gj_0 = p.sum_table(np.multiply(A[0,:], B[0,:]))
        gj_1 = p.sum_table(np.multiply(A[1,:], B[1,:]))
        gj_2 = p.sum_table(np.multiply(2 * A[1,:] - A[0,:], 2 * B[1,:] - B[0,:]))    
        t_new = time.time()
        prover_time = prover_time + t_new - t_old        
        # --------------------------------------------
        # Verifier: 
        t_old = time.time()
        if 2 * (gj_0 + gj_1) != p.twice_single_var_eval(g0, g1, g2, a[i]):
            print("Error 2!", i)
            f = 1
        g0 = gj_0
        g1 = gj_1
        g2 = gj_2
        t_new = time.time()
        verifier_time = verifier_time + t_new - t_old         
    # --------------------------------------------
    # Prover:
    t_old = time.time()
    A = p.squeeze_table_l(A, a[l - 2])
    B = p.squeeze_table_l(B, a[l - 2])
    gj_0 = np.multiply(A[0], B[0])
    gj_1 = np.multiply(A[1], B[1])
    gj_2 = np.multiply(2 * A[1] - A[0], 2 * B[1] - B[0])
    t_new = time.time()
    prover_time = prover_time + t_new - t_old     
    # --------------------------------------------
    # Verifier: 
    t_old = time.time()
    if 2 * (gj_0 + gj_1) != p.twice_single_var_eval(g0, g1, g2, a[l - 2]):
        print("Error 3!", i)
        print(p.twice_single_var_eval(g0, g1, g2, a[l - 2]))
        f = 1
    g0 = gj_0
    g1 = gj_1
    g2 = gj_2
    # --------------------------------------------
    # Verifier: 
    if p.twice_single_var_eval(g0, g1, g2, a[l - 1]) != 2 * s:
        print("Error 4!")
        f = 1
    t_new = time.time()
    verifier_time = verifier_time + t_new - t_old
    if f == 0:
        print("Done!")    
    return prover_main_load, prover_time, verifier_time   