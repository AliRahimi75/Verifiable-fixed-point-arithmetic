import time
import random 



########## 
##################### 
################################# 
############################################# 
######################################################### Aggregate range proof: 
# computes e^b % prime_number2 where e belongs to the group 
def power(e, b, p): 
    pow_e = [] 
    pow_e.append(e)  
    c = 1  
    b = bin(b)
    if b[0] == "-":
        print("The power is negative")
        return 
    b = b[2:]
    b = b[::-1]

    for i in range(len(b)):  
        if b[i] == "1":
            c = c * pow_e[i] % p  
        aux = pow_e[i] * pow_e[i]
        aux = aux % p   
        pow_e.append(aux) 
    return c 

def commit(g, u, p): # p usually is the big prime number
    c = power(g[0], u[0], p)  
    for i in range(1, len(u)):
        c = (c * power(g[i], u[i], p)) % p
    return c

def innerproduct(a, b, p):
    c = a[0] * b[0] % p
    for i in range(1, len(a)):
        c = (c + a[i] * b[i]) % p
    return c 
  
def inv(a, p):
    return power(a, p - 2, p) 

def vector_to_element(v, m, p):
    u = []
    for i in range(len(v)):
        u.append(power(v[i], m, p))
    return u

def hadamard(f, t, p):
    g = []
    for i in range((len(f))):
        g.append(f[i] * t[i] % p)
    return g

def vector_of_powers(a, n, p):
    result = []
    k = 1
    for i in range(n-1):
        result.append(k) 
        k = k * a % p 
    result.append(k) 
    return result     

def sum_of_vector(a, p):
    b = 0
    for i in range(len(a)):
        b = (b + a[i]) % p 
    return b

def change_base(h, y, order, p):
    g = []
    y_inv = inv(y, order)
    for i in range(len(h)):
        g.append(power(h[i], power(y_inv, i, order), p))
    return g

def element_exp_number(g, z, p):
    b = 1
    for i in range(len(g)):
        b = b * power(g[i], z, p) % p 
    return b


##########
#####################
#################################
#############################################
######################################################### SUM-CHECK functions:
def mod_vector(a, p):
    return [a[i] % p for i in range(len(a))] 

def binary_reshape_renov(A, p):
    if type(A) == list and type(A[0]) == list and A[0][0] != list: # Ensure A is a matrix of the form [[...],...,[...]] 
        D = []
        for i in range(len(A)): 
            D = D + mod_vector(A[i], p) 
        return D
    else:
        print("Tha matrix has not an appropriate shape!")


def matmul_renov(A, B, p): 
    s1 = len(A) 
    s2 = len(A[0]) 
    s3 = len(B) 
    s4 = len(B[0]) 
    if s2 != s3:
        print("Matrices are miss matched!")
        return 0
    D = []
    for i in range(s1):
        row = []
        for j in range(s4): 
            a = 0
#            print("type(a) = ", type(a))
            for k in range(s2):
                a = (a + A[i][k] * B[k][j]) % p 
#                print("type(a) = ", type(a))
            row.append(a) 
        D = D + [row] 
    return D


def MLE_vector_renov(r, p): 
    l = len(r) 
    if l == 1: 
        return [(1-r[0]) % p, r[0] % p] 
    else: 
        a = MLE_vector_renov(r[1:], p) 
        b = [(1-r[0]) * a[i] % p for i in range(len(a))] 
        c = [r[0] * a[i] % p for i in range(len(a))] 
        d = b + c 
        return d 


def MLE_renov(C, r, p):
    # C(1,0,0,...,1) is the given values of a map on the unit multi dimension cube. 
    # We want to compute MLE of C on the point r, that is \tilde{C}(r_1,r_2,...,r_l) 
    # It can be evaluated by a vector multiplication 
    v = MLE_vector_renov(r, p)  
    r = 0 
    for i in range(len(C)):
        r = (r + C[i] * v[i]) % p 
    return r 

def MLE_renov_time_version(C, r, order, prime_number, factors, base_group): 
    N = len(C) 
    Gen = generators(2 * N + 1, prime_number, factors, base_group)     # Already prepared in action 
    t_old = time.time()
    v = MLE_vector_renov(r, order)   
    inner_prod = innerproduct(C, v, order)     
    P = commit(Gen, C + v + [inner_prod], prime_number)  
    t_new = time.time() 
    t1 = t_new - t_old 
    t_total = innerproduct_argument_time_version(list(C), list(v), P, Gen, order, prime_number)  
    # innerproduct_argument_print_version(a, b, P, Gen)  
    return [inner_prod, t_total[0] + t1, t_total[1], 2 + t_total[2]]  # 2 + t_total[2] is the communication where 2 is for prime selection and commitment


def sum_table_renov(G, p): # tables of dimension (2, 2, 2, ...)
    b = 0
    for i in range(len(G)):
        b = (b + G[i]) % p
    return b


def squeeze_table_l_renov(G, r1, p): 
    l = int(len(G) / 2)
    G_new = []
    for k in range(l): 
        a = G[k] * (1 - r1) % p 
        b = G[l + k] * r1 % p
        G_new.append((a + b) % p)
    return G_new 


def squeeze_table_r_renov(G, r1, p): 
    l = int(len(G) / 2)
    G_new = []
    for k in range(l): 
        a = G[2 * k] * (1 - r1) % p 
        b = G[2 * k + 1] * r1 % p
        G_new.append((a + b) % p)
    return G_new 


def twice_single_var_eval_renov(g0, g1, g2, t, p): 
    aux1 = (g0 + -2*g1 % p + g2) % p  
    aux1 = (t**2 % p) * aux1 % p 
    aux2 = (-3*g0 % p + 4*g1 % p + -g2 % p) % p 
    aux2 = t * aux2 % p 
    return (aux1 + aux2 + 2*g0 % p) % p 


##########
#####################
#################################
#############################################
######################################################### Setup:
# ------------------------------------------------------------------- 
# Set the vector of generators 
# This function finds gen_num generators of the finite field.
def factor(gen_num, p_n, factors, base_group):
    # factors = [2, 3, 5, 17, 257, 65537]  # prime factors of the order. Each generator and each number is this set must be co-prime.
    k = 0
    for i in range(int(p_n / 10), p_n):
        flag = 1
        for j in factors:
            if power(i, j, p_n) == 1:  
                flag = 0
                break
        if flag == 1:
            g = i 
            k = k + 1  
        if k >= 1:
            break

    generators = [0 for i in range(gen_num)]
    # b = 2**32 * 3 * 5 * 17 * 257 
    for i in range(1, gen_num + 1): # 1 is in the group but is not a generator 
        generators[i - 1] = power(g, base_group * i, p_n) 
    return generators 


def generators(N, pn, factors, base_group):
    return factor(2 * N + 1, pn, factors, base_group)


##########
#####################
#################################
#############################################
######################################################### Inner product argument:

def innerproduct_argument_time_version(a, b, P, Gen, order, prime_number): 
    comm = 0  
    prover_time = 0 
    verifier_time = 0 
    t1 = time.time()
    random_range = 100
    n = int(len(a)/2) 
    z = [0 for i in range(n)] 
    L = commit(Gen, z + a[:n] + b[n:] + z + [innerproduct(a[:n], b[n:], order)], prime_number) 
    R = commit(Gen, a[n:] + z + z + b[:n] + [innerproduct(a[n:], b[:n], order)], prime_number) 
    t2 = time.time()
    prover_time = prover_time + t2 - t1 
    comm = comm + 2 
#    print("Prover sends L =", L, "and R =", R, "to Verifier.")  

    t1 = time.time()
    x = random.randint(1, random_range)
    comm = comm + 1 
#    print("Verifier sends x =", x ,"to Prover.") 

    x_m = inv(x, order) 
    x_2 = power(x, 2, order)  
    x_m2 = power(x_m, 2, order)  

    a = [(x * a[i] + x_m * a[n+i]) % order for i in range(n)]
    b = [(x_m * b[i] + x * b[n+i]) % order for i in range(n)]

    P_left = ((power(L, x_2, prime_number) * P % prime_number) * power(R, x_m2, prime_number)) % prime_number 
    Gen_new = hadamard(vector_to_element(Gen[:n], x_m, prime_number), vector_to_element(Gen[n:2*n], x, prime_number), prime_number) + hadamard(vector_to_element(Gen[2*n:3*n], x, prime_number), vector_to_element(Gen[3*n:4*n], x_m, prime_number), prime_number) + [Gen[4*n]] 
              
    t2 = time.time() 
    verifier_time = verifier_time + t2 - t1 
#    print("Verifier updates P and Generators.") 

    if n == 1: 
        P_right = commit(Gen_new, a + b + [innerproduct(a, b, order)], prime_number)   
        comm = comm + 2
        if P_left == P_right: 
            return [prover_time, verifier_time, comm]  
        else: 
            print("P_left != P_right") 
    else: 
        try:
            aux1 = [prover_time, verifier_time, comm] 
            aux2 = innerproduct_argument_time_version(a, b, P_left, Gen_new, order, prime_number)
            return [aux1[0] + aux2[0], aux1[1] + aux2[1], aux1[2] + aux2[2]]  # communication is aux1[2] + aux2[2] 
        except:
            print("An error ocerred in innerproduct!")



##########
#####################
#################################
#############################################
######################################################### Main function:
def matmul_list(a, b, p, bias):  
    c = []  
    if len(a[0]) != len(b):  
        print("Error: Dimension mismatch!")  
    for i in range(len(a)):  
        r = []  
        for j in range(len(b[0])):  
            d = 0  
            for k in range(len(a[0])):  
                d = ((d + a[i][k] * b[k][j] + bias) % p) - bias  
            r.append(d)  
        c.append(r)  
    return c  

def error_vector(a, precision): 
    b = [] 
    d = 2**(precision-1)
    e = 2 * d
    for i in range(len(a)):
        c = []
        for j in range(len(a[0])):
            c.append(((a[i][j] + d) % e) - d)
        b.append(c)
    return b 

def commit_matrix(g, u, p): # p usually is the big prime number
    return commit(g, u[0], p)


def round_vector(a, e, precision, bias, p): 
    b = [] 
    for i in range(len(a[0])): 
        b.append(((((a[0][i] - e[0][i]) >> precision) + bias) % p) - bias)   
    return [b] 

def activcation_quadratic(a, bias, p):
    b = []
    for i in range(len(a[0])):
        b.append(((a[0][i]**2 + bias) % p) - bias)
    return [b] 
    
def identity(a):
    zero_vector = [0 for i in range(a)]
    c = [zero_vector for i in range(a)] 
    for i in range(a):
        c[i][i] = 1
    return c 

def prime_order(item):
    match item:
        case 1:
            prime_number = 58498015517779011107  # 66 bits
            order = 10412605111744217  # 54 bits
            factors = [2, 53, 10412605111744217]  # factors of (prime_number - 1) 
            base_group = 2 * 53**2  # base_group * order = prime_number - 1
        case 2: 
            prime_number = 3413338781 # 32 bits
            order = 2160341 # 22 bits
            factors = [2, 5, 79, 2160341]  # factors of (prime_number - 1)
            base_group = 2**2 * 5 * 79  # base_group * order = prime_number - 1
        case 3:
            prime_number = 634424966586249095481352779403952798173 # 129 bits 
            order = 7552678173645822565254199754808961883 # 123 bits 
            factors = [2, 3, 7, 7552678173645822565254199754808961883]  # factors of (prime_number - 1) 
            base_group = 2**2 * 3 * 7  # base_group * order = prime_number - 1
        case 4: 
            prime_number = 21163 # 15 bits 
            order = 3527 # 12 bits 
            factors = [2, 3, 3527]  # factors of (prime_number - 1) / order
            base_group = 2 * 3  # base_group * order = prime_number - 1              
        case 5: 
            prime_number = 7943623187295221 # 53 bits 
            order = 397181159364761 # 49 bits 
            factors = [2, 5, 397181159364761]  # factors of (prime_number - 1) / order
            base_group = 2**2 * 5  # base_group * order = prime_number - 1    
        case 6: 
            prime_number = 656729620295482933195950642396373 # 110 bits 
            order = 411484724495916624809492883707 # 99 bits 
            factors = [2, 3, 7, 19, 411484724495916624809492883707]  # factors of (prime_number - 1) / order
            base_group = 2**2 * 3 * 7 * 19 # base_group * order = prime_number - 1                              
        case 7: 
            prime_number = 596296757809853978047039229 # 89 bits 
            order = 149074189452463494511759807 # 87 bits 
            factors = [2, 149074189452463494511759807]  # factors of (prime_number - 1) / order
            base_group = 2**2  # base_group * order = prime_number - 1  
        case 8: 
            prime_number = 2186526012123210939319 # 71 bits 
            order = 364421002020535156553 # 69 bits 
            factors = [2, 3, 364421002020535156553]  # factors of (prime_number - 1) / order
            base_group = 2*3  # base_group * order = prime_number - 1 
        case 9: 
            prime_number = 12329774216320990153 # 64 bits 
            order = 9693218723522791 # 54 bits 
            factors = [2, 3, 53, 9693218723522791]  # factors of (prime_number - 1) / order
            base_group = 2**3 * 3 * 53 # base_group * order = prime_number - 1 
        case 10:
            prime_number = 32823477210247 # 45 bits 
            order = 5470579535041 # 43 bits 
            factors = [2, 3, 5470579535041]  # factors of (prime_number - 1) / order
            base_group = 2 * 3 # base_group * order = prime_number - 1         
        case 11:
            prime_number = 155116993873 # 38 bits 
            order = 3231604039 # 32 bits 
            factors = [2, 3, 3231604039]  # factors of (prime_number - 1) / order
            base_group = 2**4 * 3 # base_group * order = prime_number - 1  
        case 12:
            prime_number = 1246303 # 21 bits 
            order = 69239 # 17 bits 
            factors = [2, 3, 69239]  # factors of (prime_number - 1) / order
            base_group = 2 * 3**2 # base_group * order = prime_number - 1  
        case 13:
            prime_number = 64330483507341957024928066854873238610009696896063 # 166 bits 
            order = 744100718385985113759086529887260723737591053 # 150 bits 
            factors = [2, 3, 1601, 744100718385985113759086529887260723737591053]  # factors of (prime_number - 1) / order
            base_group = 2 * 3**3 * 1601 # base_group * order = prime_number - 1    
        case 14:
            prime_number = 440878079 # 29 bits 
            order = 220439039 # 28 bits 
            factors = [2, 220439039]  # factors of (prime_number - 1) / order
            base_group = 2 # base_group * order = prime_number - 1                  
        case 15:
            prime_number = 45686602649 # 36 bits 
            order = 5710825331 # 33 bits 
            factors = [2, 5710825331]  # factors of (prime_number - 1) / order
            base_group = 2**3 # base_group * order = prime_number - 1         
        case 16:
            prime_number = 29780579 # 25 bits 
            order = 14890289 # 24 bits 
            factors = [2, 14890289]  # factors of (prime_number - 1) / order
            base_group = 2 # base_group * order = prime_number - 1   
        case 17:
            prime_number = 120124259 # 27 bits 
            order = 60062129 # 26 bits 
            factors = [2, 60062129]  # factors of (prime_number - 1) / order
            base_group = 2 # base_group * order = prime_number - 1  
        case 18:
            prime_number = 53087208899116140505086771455101577892047 # 136 bits 
            order = 26543604449558070252543385727550788946023 # 135 bits 
            factors = [2, 26543604449558070252543385727550788946023]  # factors of (prime_number - 1) / order
            base_group = 2 # base_group * order = prime_number - 1  
        case 19:
            prime_number = 46500509893249816007603466927722270658960569310683617337567 # 195 bits 
            order = 71982213457043058835299484408238809069598404505702194021 # 186 bits 
            factors = [2, 17, 19, 71982213457043058835299484408238809069598404505702194021]  # factors of (prime_number - 1) / order
            base_group = 2 * 17 * 19 # base_group * order = prime_number - 1          
        case 20:
            prime_number = 839107319326348412695666714082109091434507526707256172591963251 # 210 bits 
            order = 58884724163252520189169593970674322205930352751386398076629 # 196 bits 
            factors = [2, 3, 5, 19, 58884724163252520189169593970674322205930352751386398076629]  # factors of (prime_number - 1) / order
            base_group = 2 * 3 * 5**3 * 19 # base_group * order = prime_number - 1  


    # prime number generator website = https://bigprimes.org/ 
    # large number factorization = https://www.numberempire.com/numberfactorizer.php 
    return [prime_number, order, factors, base_group]


