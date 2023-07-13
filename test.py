import numpy as np
import MatMul as MatMul
import matplotlib.pyplot as plt

Vector_n = np.array([])
Vector_p_time = np.array([])
Vector_p_main_load = np.array([])
Vector_v_time = np.array([])
for i in range(8):
    n = 8 * 2**i 
    m = 8 * 2**i 
    o = 8 * 2**i
    x = 10
    A = np.random.rand(n * m).reshape(n,m) 
    A = np.ceil(A * x)
    B = np.random.rand(m * o).reshape(m,o)
    B = np.ceil(B * x)
    [p_main_load, p_time, v_time] = MatMul.Thaler_method(A,B)
    # print(p_time, v_time)
    
    Vector_n = np.concatenate((Vector_n, [n]))
    Vector_p_time = np.concatenate((Vector_p_time, [p_time]))
    Vector_p_main_load = np.concatenate((Vector_p_main_load, [p_main_load]))
    Vector_v_time = np.concatenate((Vector_v_time, [v_time]))
    """
    plt.plot(n, p_time, 'o')
    plt.plot(n, p_main_load, '.')
    plt.plot(n, v_time, '*')
    """

plt.plot(Vector_n, Vector_p_time)
plt.plot(Vector_n, Vector_p_main_load)
plt.plot(Vector_n, Vector_v_time)

plt.show()

