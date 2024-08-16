import time 
import Useful_functions as aux 
import random 
import math 


def Thaler_method(A, B, p, prime_number, factors, base_group): 
    prover_time = 0 
    verifier_time = 0 
    MLE_v_time_GKR = 0
    MLE_p_time_GKR = 0
    MLE_v_time = 0
    MLE_p_time = 0    
    comm = 0 # The number of finite field or group element sent in either direction 
    # --------------------------------------------
    # Setup: given two matrices A and B
    n = len(A) # must be a power of two and at least 8
    m = len(A[0]) # must be a power of two and at least 8
    o = len(B[0]) # must be a power of two and at least 8
#     if n < 8 or m < 8 or o < 8:
#         print("Warning: Matrices are not large at least in one dimension!")
    if math.log2(n) - int(math.log2(n)) != 0:
        print("Sumcheck Error! The first dimension of A is not a power of 2")
    if math.log2(m) - int(math.log2(m)) != 0:
        print("Sumcheck Error! The second dimension of A is not a power of 2") 
    if math.log2(o) - int(math.log2(o)) != 0:
        print("Sumcheck Error! The first dimension of B is not a power of 2")        
    if m != len(B):
        print("Sumcheck Error! A and B cannot be multipied due to dimension mismatch!")                      
    f_renov = 0
    x = 100 # Precision 
    # -------------------------------------------- 
    # Prover: I've computed A.B correctly. 
    t_old = time.time() 
    C_renov = aux.binary_reshape_renov(aux.matmul_renov(A, B, p), p)   
    A_renov = aux.binary_reshape_renov(A, p)  
    B_renov = aux.binary_reshape_renov(B, p)  
    t_new = time.time() 
    prover_time = prover_time + t_new - t_old
    # --------------------------------------------
    # Verifier: Here you are: r_1 and r_2. Give me \tilde{f}_C(r_1,r_2)
    t_old = time.time() 
    r_1 = [random.randint(1, x) for i in range(int(math.log2(n)))]     
    r_2 = [random.randint(1, x) for i in range(int(math.log2(o)))]   
    comm = comm + int(math.log2(n)) + int(math.log2(o)) # Sent from Verifier to Prover
    # -------------------------------------------- Start sumcheck 
    # Verifier and Prover: 
    l = int(math.log2(m)) 
    a = [random.randint(1, x) for i in range(l)]  
    t_new = time.time() 
    verifier_time = verifier_time + t_new - t_old   
    # s_renov = aux.MLE_renov(A_renov, r_1 + a, p) * aux.MLE_renov(B_renov, a + r_2, p) % p  
    yy1 = aux.MLE_renov_time_version(A_renov, r_1 + a, p, prime_number, factors, base_group)  
    yy2 = aux.MLE_renov_time_version(B_renov, a + r_2, p, prime_number, factors, base_group)  
    MLE_v_time = MLE_v_time + yy1[2] + yy2[2]  
    MLE_p_time = MLE_p_time + yy1[1] + yy2[1]  
    MLE_v_time_GKR = MLE_v_time_GKR + yy2[2] 
    MLE_p_time_GKR = MLE_p_time_GKR + yy2[1]     
    s_renov = yy1[0] * yy2[0] % p 
    comm = comm + yy1[3] + yy2[3] 
    # -------------------------------------------- 
    # C1_renov = aux.MLE_renov(C_renov, r_1 + r_2, p) 
    yy3 = aux.MLE_renov_time_version(C_renov, r_1 + r_2, p, prime_number, factors, base_group)    
    MLE_v_time = MLE_v_time + yy3[2]  
    MLE_p_time = MLE_p_time + yy3[1]  
    C1_renov = yy3[0] 
    comm = comm + yy3[3]
    # Prover: 
    t_old = time.time() 
    for i in range(int(math.log2(n))): 
        A_renov = aux.squeeze_table_l_renov(A_renov, r_1[i], p) 
    for i in reversed(range(int(math.log2(o)))): 
        B_renov = aux.squeeze_table_r_renov(B_renov, r_2[i], p)        

    len_A_renov = int(len(A_renov) / 2) 
    gj_0_renov = aux.sum_table_renov([A_renov[i] * B_renov[i] % p for i in range(len_A_renov)], p)   
    gj_1_renov = aux.sum_table_renov([A_renov[i] * B_renov[i] % p for i in range(len_A_renov, 2 * len_A_renov)], p)   
    gj_2_renov = aux.sum_table_renov([((2 * A_renov[len_A_renov + i] - A_renov[i]) % p) * ((2 * B_renov[len_A_renov + i] - B_renov[i]) % p) % p for i in range(len_A_renov)], p)   
    t_new = time.time()
    prover_time = prover_time + t_new - t_old 
    comm = comm + 3 # 3 is for gj_0_renov, gj_1_renov, gj_2_renov 
    # --------------------------------------------
    # Verifier:
    t_old = time.time()
    if (gj_0_renov + gj_1_renov) % p != C1_renov:
        print("Error 1_renov!")
        f_renov = 1        
    g0_renov = gj_0_renov
    g1_renov = gj_1_renov
    g2_renov = gj_2_renov    
    t_new = time.time()
    verifier_time = verifier_time + t_new - t_old    
    # --------------------------------------------
    for i in range(l - 1):
        # Prover:
        t_old = time.time()         
        A_renov = aux.squeeze_table_l_renov(A_renov, a[i], p)         
        B_renov = aux.squeeze_table_l_renov(B_renov, a[i], p)     
        comm = comm + 1 # For a[i] sent   
        len_A_renov = int(len(A_renov) / 2) 
        gj_0_renov = aux.sum_table_renov([A_renov[i] * B_renov[i] % p for i in range(len_A_renov)], p)         
        gj_1_renov = aux.sum_table_renov([A_renov[i] * B_renov[i] % p for i in range(len_A_renov, 2 * len_A_renov)], p)         
        gj_2_renov = aux.sum_table_renov([((2 * A_renov[len_A_renov + i] - A_renov[i]) % p) * ((2 * B_renov[len_A_renov + i] - B_renov[i]) % p) % p for i in range(len_A_renov)], p)   
        comm = comm + 3 # 3 is for gj_0_renov, gj_1_renov, gj_2_renov 
        t_new = time.time() 
        prover_time = prover_time + t_new - t_old        
        # --------------------------------------------
        # Verifier: 
        t_old = time.time()
        PART1_renov = 2 * (gj_0_renov + gj_1_renov) % p
        PART2_renov = aux.twice_single_var_eval_renov(g0_renov, g1_renov, g2_renov, a[i], p)
        if PART1_renov != PART2_renov:
            print("Error 2_renov!", i)
            f_renov = 1            
        g0_renov = gj_0_renov
        g1_renov = gj_1_renov
        g2_renov = gj_2_renov        
        t_new = time.time()
        verifier_time = verifier_time + t_new - t_old         
    # --------------------------------------------
    # Verifier: 
    t_old = time.time() 
    PART1_renov = 2 * s_renov % p 
    PART2_renov = aux.twice_single_var_eval_renov(g0_renov, g1_renov, g2_renov, a[l - 1], p) 
    if PART1_renov != PART2_renov: 
        print("Error 4_renov!") 
        f_renov = 1         
    t_new = time.time() 
    verifier_time = verifier_time + t_new - t_old 
      
    if f_renov == 0: 
        print("Done_renov!")           
    return [MLE_v_time_GKR, MLE_p_time_GKR, MLE_v_time, MLE_p_time, prover_time, verifier_time, comm] 
