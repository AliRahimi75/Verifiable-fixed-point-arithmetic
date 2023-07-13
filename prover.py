import numpy as np

prime_number = 2^64 - 2^32 + 1

def MatMul_usual(A,B): 
    if len(A.shape) != 2:
        print("Error: Matrix A is not of dimension 2, it is of dimension", len(A.shape))
        return None
    if len(B.shape) != 2:
        print("Error: Matrix B is not of dimension 2, it is of dimension", len(B.shape))
        return None
    if A.shape[1] != B.shape[0]:
        print("Error: Matrix A cannot be multiplied by B because of dimension mismatch")
        return None    
    C = np.zeros((A.shape[0], B.shape[1]))        
    C = np.matmul(A,B)                
    return C    

def MatMul(A, B, v): 
    # Matrix multiplication of two maps which are defined over the v-dimensional unit cube.
    a = len(A.shape) - v
    b = len(B.shape) - v
    C = np.zeros(tuple([2 for i in range(a + b)]))
    for i in range(2**a): 
        for j in range(2**b): 
            for k in range(2**v): 
                bin = np.binary_repr(i, width=a) + np.binary_repr(j, width=b)
                d3  = tuple(int(i) for i in bin)   
                C[d3]
                bin = np.binary_repr(i, width=a) + np.binary_repr(k, width=v)
                d1  = tuple(int(i) for i in bin)                   
                A[d1]
                bin = np.binary_repr(k, width=v) + np.binary_repr(j, width=b)
                d2  = tuple(int(i) for i in bin)                   
                B[d2] # ?! I dont remeber why i wrote this
                C[d3] = C[d3] + A[d1] * B[d2]
    return C


def MLE_vector(l, r): 
    if l == 1:
        return np.remainder(np.array([1-r[0], r[0]]), prime_number)
    else:
        a = MLE_vector(l-1, r[1:])
        return np.remainder(np.array([(1 - r[0]) * a, r[0] * a]), prime_number)
        

def MLE(C, r): 
    # C(1,0,0,...,1) is the given values of a map on the unit multi dimension cube. 
    # We want to compute MLE of C on the point r, that is \tilde{C}(r_1,r_2,...,r_l) 
    # It can be evaluated by a vector multiplication 
    l = len(C.shape)
    v = MLE_vector(l, r)
    r = 0
    for k in range(2**l): 
        bin = np.binary_repr(k, width=l)
        d  = tuple(int(i) for i in bin)   
        r = np.remainder(r + C[d] * v[d], prime_number)
    return r

def sum_table(G): 
    return np.remainder(np.sum(G), prime_number)

def squeeze_table_l(G, r1): 
    l = len(G.shape)
    G_new = np.zeros(tuple([2 for i in range(l - 1)]))
    for k in range(2**(l-1)):
        bin = np.binary_repr(k, width=l-1)
        d  = tuple(int(i) for i in bin)   
        G_new[d] = np.remainder(G[(0,) + d] * (1 - r1) + G[(1,) + d] * r1, prime_number)
    return G_new

def squeeze_table_r(G, r1): 
    l = len(G.shape)
    G_new = np.zeros(tuple([2 for i in range(l - 1)]))
    for k in range(2**(l-1)):
        bin = np.binary_repr(k, width=l-1)
        d  = tuple(int(i) for i in bin)   
        G_new[d] = np.remainder(G[d + (0,)] * (1 - r1) + G[d + (1,)] * r1, prime_number)
    return G_new


def single_var_eval(g0, g1, g2, t): 
    return 1/2 * (t**2 * (g0 - 2*g1 + g2) + t * (-3*g0 + 4*g1 - g2) + 2*g0)

def twice_single_var_eval(g0, g1, g2, t): 
    return np.remainder(t**2 * (g0 - 2*g1 + g2) + t * (-3*g0 + 4*g1 - g2) + 2*g0, prime_number)

def quaternary_single_var_eval(g0, g1, g2, g3, t): 
    return t**3 * (-g0 + 3*g1 - 3*g2 + g3) + t**2 * (6*g0 - 15*g1 + 12*g2 -3*g3) + t * (-11*g0 + 18*g1 - 9*g2 + 2*g3) + 6*g0

def quinary_single_var_eval(g0, g1, g2, g3, g4, t):
    return np.remainder(t**4 * (g0 - 4*g1 + 6*g2 - 4*g3 + g4) + \
                        t**3 * (-10*g0 + 36*g1 - 48*g2 + 28*g3 - 6*g4) + \
                            t**2 * (35*g0 - 104*g1 + 114*g2 - 56*g3 + 11*g4) + \
                                t * (-50*g0 + 96*g1 - 72*g2 + 32*g3 - 6*g4) + 24*g0, prime_number)

