import random 
# from Innerproduct_argument import innerproduct_argument_time_version 
from Useful_functions import innerproduct_argument_time_version, inv, element_exp_number, power, commit, innerproduct, hadamard, vector_of_powers, sum_of_vector, change_base 
from Useful_functions import generators
import time 

random_range = 100

def bit_concat(m, n, v):
    a_L = []
    aux1 = 2**n
    for i in v:
        V_binary = bin(i + aux1)  # To make all binary representations of length n + 1
        V_binary = V_binary[3:]
        b = list(map(int, list(V_binary)))
        b = b[::-1] 
        a_L = a_L + b 
    return a_L 


def special_function1(h, powers_of_2, z, order, prime_number, m, n):
    a = 1 
    z_aux = z
    for j in range(1, m+1):
        z_aux = z_aux * z % order 
        a = a * power(commit(h[(j-1)*n : j*n], powers_of_2, prime_number), z_aux, prime_number) % prime_number 
    return a


def aggregated_range_proof(v, n, order, prime_number, factors, base_group): 
    comm = 0 
    # Proves all elements in v has at most n bits.
    m = len(v) 
    Gen = generators(m*n, prime_number, factors, base_group) 
    P = commit(Gen[0:m], v, prime_number) 
    comm = comm + 1 
    prover_time = 0 
    verifier_time = 0 
    y = random.randint(2, random_range) 
    z = random.randint(2, random_range) 
    comm = comm + 2 

    t1 = time.time()
    a_L = bit_concat(m, n, v)
    a_R = [(a_L[i] - 1) % order for i in range(m*n)] 
    A = commit(Gen[0:m*n], a_L, prime_number) * commit(Gen[m*n:2*m*n], a_R, prime_number) % prime_number  
    comm = comm + 1

    l = [a_L[i] - z for i in range(m*n)] 
    powers_of_y = vector_of_powers(y, m*n, order) 
    powers_of_2 = vector_of_powers(2, n, order) 

    aux2 = [0 for i in range(m*n)] 
    z_aux = z 
    for j in range(1,m+1): 
        z_aux = z_aux * z % order 
        aux2[(j-1)*n : j*n] = [powers_of_2[i] * z_aux % order for i in range(n)] 
    aux3 = hadamard(powers_of_y, [a_R[i] + z for i in range(m*n)], order)
    r = [(aux2[i] + aux3[i]) % order for i in range(m*n)] 

    powers_of_z = vector_of_powers(z, m, order) 
    q_z = innerproduct(powers_of_z, v, order) 
    comm = comm + 1 
    
    P2 = commit(Gen[m:2*m], powers_of_z, prime_number) 
    P_aux = (P * P2 % prime_number) * power(Gen[2*m], q_z, prime_number) % prime_number 
    t2 = time.time() 
    prover_time = prover_time + t2 - t1    

    t_total = innerproduct_argument_time_version(v, powers_of_z, P_aux, Gen[0:2*m+1], order, prime_number) 
#    print("Prover time (ms) = ", int(1000 * t_total[0]), " and Verifier time (ms) = ", int(1000 * t_total[1])) 
#    print("-------------------------------------------------------")
    prover_time = prover_time + t_total[0] 
    verifier_time = verifier_time + t_total[1] 
    comm = comm + t_total[2] 

    t1 = time.time()
    delta = ((z - z**2) * sum_of_vector(powers_of_y, order) - z**3 * sum_of_vector(powers_of_z, order) * sum_of_vector(powers_of_2, order)) % order 
    h_new = change_base(Gen[m*n:2*m*n], y, order, prime_number) 

    t = (delta + z**2 * q_z) % order 
    # print("innerproduct == t :::::", (innerproduct(r, l, order)) % order == t % order) 

    P_aux1 = element_exp_number(Gen[0:m*n], -z % order, prime_number) 
    P_aux2 = power(commit(h_new, powers_of_y, prime_number), z, prime_number) 
    P_aux3 = special_function1(h_new, powers_of_2, z, order, prime_number, m, n) 
    P3 = (((A * P_aux1 % prime_number) * P_aux2 % prime_number) * P_aux3) % prime_number 
    d = l + r 
    d = [d[i] % order for i in range(len(d))] 
    # print("P3_new == P3 :::::", P3 == commit(Gen[0:m*n] + h_new, d, prime_number)) 

    l = [l[i] % order for i in range(len(l))] 
    P4 = P3 * power(Gen[2*m*n], t, prime_number) % prime_number 
    t2 = time.time() 
    prover_time = prover_time + t2 - t1 
    verifier_time = verifier_time + t2 - t1 

    t_total = innerproduct_argument_time_version(l, r, P4, Gen[0:m*n] + h_new + [Gen[2*m*n]], order, prime_number) 
    prover_time = prover_time + t_total[0] 
    verifier_time = verifier_time + t_total[1]     
    comm = comm + t_total[2]  
#    print("Prover time (ms) = ", int(1000 * t_total[0]), " and Verifier time (ms) = ", int(1000 * t_total[1])) 
#    print("-------------------------------------------------------")

    return [prover_time, verifier_time, comm]



"""
# Initialization parameters:
m = 512 # The number of elements in the vector 
n = 8 # The number of bits to check each element is in the range [0, 2**n-1]
 
v = [random.randint(1, min(random_range, 2**n) - 1) for i in range(m)] 

t_total = aggregated_range_proof(v, n)        
print("m = ", m, " and n = ", n, " and Prover time (ms) = ", int(1000 * t_total[0]), " and Verifier time (ms) = ", int(1000 * t_total[1])) 
""" 





