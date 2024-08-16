import Useful_functions as aux 
from Useful_functions import generators, commit 
import SumCheck_abs_value as SumCheck_abs
from Aggregated_range_proof import aggregated_range_proof
import time

def verify_relu(C_renov, input_item, precision, mantisa, bias): 
    [prime_number, order, factors, base_group] = input_item 
    final_shape = [len(C_renov), len(C_renov[0])] 
    C_renov = aux.flatten(C_renov) 
    V_abs_1 = aux.abs(C_renov) 
    D_renov = aux.relu(C_renov, V_abs_1) 

    Gen = generators(len(C_renov), prime_number, factors, base_group) 
    c1 = commit(Gen, [C_renov[i] + bias for i in range(len(C_renov))], prime_number) 
    c2 = commit(Gen, [V_abs_1[i] + bias for i in range(len(V_abs_1))], prime_number) 
    c3 = commit(Gen, [D_renov[i] + bias for i in range(len(D_renov))], prime_number) 

    if c1 * c2 % prime_number != c3**2 % prime_number:
        print("Error in ReLU")

    [_, _, p_time1, v_time1, comm1] = SumCheck_abs.equality_check([C_renov], [V_abs_1], order, prime_number, factors, base_group) # Check C_renov**2 = V_abs_1**2 


    [p_time2, v_time2, comm2] = aggregated_range_proof([V_abs_1[i] for i in range(len(V_abs_1))], precision + mantisa + 2, order, prime_number, factors, base_group) 

    main_time = 0.008 # Obtained from the file "matrix NN_v2.py"

    return main_time, p_time1 + p_time2, v_time1 + v_time2, comm1 + comm2, aux.list_reshape(D_renov, final_shape[0], final_shape[1]) 


# [prime_number, order, factors, base_group] = aux.prime_order(10) 
# print(verify_relu([[12, -4], [5, -10]], [prime_number, order, factors, base_group], 0, 0, 50))

# MatMul.Thaler_method(A, B, p, prime_number, factors, base_group)  


