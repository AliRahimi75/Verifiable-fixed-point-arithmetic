import numpy as np
def binary_reshape(A):
    a = A.shape[0]
    b = A.shape[1]
    r = int(np.log2(a))
    v = int(np.log2(b))
    A_new = np.zeros(tuple([2 for i in range(np.log2(A.shape[0] * A.shape[1]).astype(int))]))
    if v > 0 and r > 0:
        for i in range(a):
            for j in range(b):
                bin = np.binary_repr(i, width=r) + np.binary_repr(j, width=v)
                d  = tuple(int(i) for i in bin)
                A_new[d] = A[i,j]
        return A_new
    if v > 0 and r == 0:
        for i in range(a):
            for j in range(b):
                bin = np.binary_repr(j, width=v)
                d  = tuple(int(i) for i in bin)
                A_new[d] = A[i,j]
        return A_new      
    if v == 0 and r > 0:
        for i in range(a):
            for j in range(b):
                bin = np.binary_repr(i, width=r)
                d  = tuple(int(i) for i in bin)
                A_new[d] = A[i,j]
        return A_new   
    if v == 0 and r == 0:
        A_new = A[0,0]
        return A_new      


def binary_reshape_vector(e):
    a = e.shape[0]
    r = int(np.log2(a))
    e_new = np.zeros(tuple([2 for i in range(np.log2(a).astype(int))]))
  
    if r > 0:
        for i in range(a):
            bin = np.binary_repr(i, width=r)
            d  = tuple(int(i) for i in bin)
            e_new[d] = e[i]
        return e_new   
    if r == 0:
        e_new = e[0,0]
        return e_new    
