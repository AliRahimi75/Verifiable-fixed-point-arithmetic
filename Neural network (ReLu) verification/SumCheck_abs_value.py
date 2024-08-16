import time 
import Useful_functions as aux 
import random 
import math 


def equality_check(A, B, p, prime_number, factors, base_group): 
    prover_time = 0 
    verifier_time = 0 
    MLE_v_time = 0 
    MLE_p_time = 0    
    comm = 0 # The number of finite field or group element sent in either direction 
    # --------------------------------------------
    # Setup: given two matrices A and B
    n = len(A[0]) # must be a power of two and at least 8 
    if math.log2(n) - int(math.log2(n)) != 0:
        print("Sumcheck Error! The first dimension of A is not a power of 2")               
    f_renov = 0 
    x = 100 # Precision 
    # -------------------------------------------- 
    # Prover: I've computed A.B correctly. 
    t_old = time.time()  
    A_renov = aux.binary_reshape_renov(A, p)  
    B_renov = aux.binary_reshape_renov(B, p)  
    I_renov = aux.binary_reshape_renov(aux.identity_matrix(n), p)  
    t_new = time.time() 
    prover_time = prover_time + t_new - t_old 
    # --------------------------------------------
    # Verifier: Here you are: r_1 and r_2. Give me \tilde{f}_C(r_1,r_2)
    t_old = time.time() 
    r_1 = [random.randint(1, x) for i in range(int(math.log2(n)))]     
    comm = comm + int(math.log2(n)) # Sent from Verifier to Prover 
    # -------------------------------------------- Start sumcheck 
    # Verifier and Prover: 
    l = int(math.log2(n)) 
    a = [random.randint(1, x) for i in range(l)]  
    t_new = time.time() 
    verifier_time = verifier_time + t_new - t_old   
    yy1 = aux.MLE_renov_time_version(I_renov, r_1 + a, p, prime_number, factors, base_group)  
    yy2 = aux.MLE_renov_time_version(A_renov, a, p, prime_number, factors, base_group)  
    yy3 = aux.MLE_renov_time_version(B_renov, a, p, prime_number, factors, base_group)  
    MLE_v_time = MLE_v_time + yy1[2] + yy2[2] + yy3[2]     
    MLE_p_time = MLE_p_time + yy1[1] + yy2[1] + yy3[1]     
    s_renov = yy1[0] * (yy2[0]**2 % p - yy3[0]**2 % p) % p 
    comm = comm + yy1[3] + yy2[3] + yy3[3] 
    # -------------------------------------------- 
    # Prover: 
    t_old = time.time() 
    for i in range(int(math.log2(n))): 
        I_renov = aux.squeeze_table_l_renov(I_renov, r_1[i], p) 

    len_A_renov = int(len(A_renov) / 2) 
    gj_0_renov = aux.sum_table_renov([I_renov[i] * (A_renov[i]**2 % p - B_renov[i]**2 % p) % p for i in range(len_A_renov)], p)   
    gj_1_renov = aux.sum_table_renov([I_renov[i] * (A_renov[i]**2 % p - B_renov[i]**2 % p) % p for i in range(len_A_renov, 2 * len_A_renov)], p)   
    gj_2_renov = aux.sum_table_renov([((2 * I_renov[len_A_renov + i] - I_renov[i]) % p) * (((2 * A_renov[len_A_renov + i] - A_renov[i]) % p)**2 % p - ((2 * B_renov[len_A_renov + i] - B_renov[i]) % p)**2 % p) % p for i in range(len_A_renov)], p)   
    gj_3_renov = aux.sum_table_renov([((3 * I_renov[len_A_renov + i] - 2 * I_renov[i]) % p) * (((3 * A_renov[len_A_renov + i] - 2 * A_renov[i]) % p)**2 % p - ((3 * B_renov[len_A_renov + i] - 2 * B_renov[i]) % p)**2 % p) % p for i in range(len_A_renov)], p)   

    t_new = time.time()
    prover_time = prover_time + t_new - t_old 
    comm = comm + 3 # 3 is for gj_0_renov, gj_1_renov, gj_2_renov 
    # --------------------------------------------
    # Verifier:
    t_old = time.time()
    if (gj_0_renov + gj_1_renov) % p != 0:
        print("Error 1_renov!")
        f_renov = 1        
    g0_renov = gj_0_renov
    g1_renov = gj_1_renov
    g2_renov = gj_2_renov    
    g3_renov = gj_3_renov    
    t_new = time.time()
    verifier_time = verifier_time + t_new - t_old    
    # --------------------------------------------
    for i in range(l - 1):
        # Prover:
        t_old = time.time()         
        A_renov = aux.squeeze_table_l_renov(A_renov, a[i], p)         
        B_renov = aux.squeeze_table_l_renov(B_renov, a[i], p)     
        I_renov = aux.squeeze_table_l_renov(I_renov, a[i], p)  
        comm = comm + 1 # For a[i] sent 
        len_A_renov = int(len(A_renov) / 2) 
        gj_0_renov = aux.sum_table_renov([I_renov[i] * (A_renov[i]**2 % p - B_renov[i]**2 % p) % p for i in range(len_A_renov)], p)   
        gj_1_renov = aux.sum_table_renov([I_renov[i] * (A_renov[i]**2 % p - B_renov[i]**2 % p) % p for i in range(len_A_renov, 2 * len_A_renov)], p)   
        gj_2_renov = aux.sum_table_renov([((2 * I_renov[len_A_renov + i] - I_renov[i]) % p) * (((2 * A_renov[len_A_renov + i] - A_renov[i]) % p)**2 % p - ((2 * B_renov[len_A_renov + i] - B_renov[i]) % p)**2 % p) % p for i in range(len_A_renov)], p)   
        gj_3_renov = aux.sum_table_renov([((3 * I_renov[len_A_renov + i] - 2 * I_renov[i]) % p) * (((3 * A_renov[len_A_renov + i] - 2 * A_renov[i]) % p)**2 % p - ((3 * B_renov[len_A_renov + i] - 2 * B_renov[i]) % p)**2 % p) % p for i in range(len_A_renov)], p)   
        comm = comm + 4 # 4 is for gj_0_renov, gj_1_renov, gj_2_renov, gj_3_renov 
        t_new = time.time() 
        prover_time = prover_time + t_new - t_old        
        # --------------------------------------------
        # Verifier: 
        t_old = time.time()
        PART1_renov = 6 * (gj_0_renov + gj_1_renov) % p 
        PART2_renov = aux.senary_single_var_eval(g0_renov, g1_renov, g2_renov, g3_renov, a[i], p)
        if PART1_renov != PART2_renov: 
            print("Error 2_renov!", i) 
            f_renov = 1            
        g0_renov = gj_0_renov
        g1_renov = gj_1_renov
        g2_renov = gj_2_renov  
        g3_renov = gj_3_renov      
        t_new = time.time()
        verifier_time = verifier_time + t_new - t_old         
    # --------------------------------------------
    # Verifier: 
    t_old = time.time() 
    PART1_renov = 6 * s_renov % p 
    PART2_renov = aux.senary_single_var_eval(g0_renov, g1_renov, g2_renov, g3_renov, a[l - 1], p)
    if PART1_renov != PART2_renov: 
        print("Error 2_renov!", i) 
        f_renov = 1       
    t_new = time.time() 
    verifier_time = verifier_time + t_new - t_old 
      
    if f_renov == 0: 
        print("Done_renov!")           
    return [MLE_v_time, MLE_p_time, prover_time, verifier_time, comm] 

