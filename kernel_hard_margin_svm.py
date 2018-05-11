import numpy as np 
import cvxopt
from cvxopt import matrix

x = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
y = [-1, -1, -1, 1, 1, 1, 1]

def kernel(x_1, x_2):
    return np.power((1 + 2*np.dot(x_1, x_2)),2)


#1 設定Q,P,A,c
def Q(x,y):
    Q_D = np.arange(49).reshape(7,7)
    for i in range(7):
        for j in range(7):
            Q_D[i,j] = y[i]*y[j]*kernel(x[i],x[j])
    return Q_D

Q_D = matrix(Q(x,y), tc='d')
p = matrix(-1*np.ones(len(y)), tc='d')
G = matrix(-1*np.eye(7), tc='d')  
h = matrix(np.zeros(len(y)), tc='d')
A = matrix([[-1], [-1], [-1], [1], [1], [1], [1]], tc='d')
c = matrix([0], tc='d')

#2 altha = QP(Q,P,A,c)
def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    return np.array(sol['x']).reshape((len(q),))

altha = cvxopt_solve_qp(Q_D, p, G, h, A, c)
print(altha)

#3 b = y_s - SUM(altha_n * y_n * K(x_n,x_s)), SV = (x_s,y_s)

#4 return SVs and their altha_n and b
#g_svm(x) = sign(SUM(altha_n * y_n * K(x_n,x))+b)
