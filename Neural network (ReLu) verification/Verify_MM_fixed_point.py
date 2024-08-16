import time
import Useful_functions as aux
import MatMul_renov as MatMul 
from Aggregated_range_proof import aggregated_range_proof 
import numpy as np 

def verify_matrix_multiplication(input_renov, A_renov, input_item, precision, mantisa):
    [prime_number, order, factors, base_group] = input_item   
    bias = int((order - 1)/2)


    t1 = time.time() 
    B_renov = aux.matmul_list(input_renov, A_renov, order, bias) 
    V_error_1 = aux.error_vector(B_renov, precision) 

    C_renov = aux.round_vector(B_renov, V_error_1, precision, bias, order) 


    t2 = time.time() 
    main_time = t2 - t1


    [MLE_v_time_GKR, MLE_p_time_GKR, MLE_v_time1, MLE_p_time1, p_time1, v_time1, comm1, comm_GKR] = MatMul.Thaler_method(input_renov, A_renov, order, prime_number, factors, base_group) 
    V_error_1 = aux.flatten(V_error_1) 
    C_renov = aux.flatten(C_renov) 

    [p_time2, v_time2, comm2] = aggregated_range_proof([V_error_1[i] + 2**(precision-1) for i in range(len(V_error_1))], precision, order, prime_number, factors, base_group) 
    [p_time3, v_time3, comm3] = aggregated_range_proof([C_renov[i] + 2**(mantisa+1) for i in range(len(C_renov))], mantisa+2, order, prime_number, factors, base_group)  

    return main_time, p_time1 + p_time2 + p_time3, v_time1 + v_time2 + v_time3, comm1 + comm2 + comm3, aux.list_reshape(C_renov, len(input_renov), len(A_renov[0]))   


