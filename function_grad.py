import numpy as np
from scipy.fft import dctn, idctn

#########################################################
################# Operator + MC + Conjugate ###################
#########################################################

# efficient gradient calculation without matrix inversion
def nll_grad_operator_MC_CGD(x, y_mul, aperture, std_z, num_ite_MC):

    print('gradient 1st term', 'K=', num_ite_MC)
    num_ite_MC_1 = num_ite_MC
    DMD_u_hat_v_mean = np.zeros_like(x)
    for MC_ite in range(num_ite_MC_1):
        v = np.random.normal(0, 1, size=np.shape(x))
        Av = A_operator(v, aperture)
        # conjugate gd
        u = conjugate_gradient(x, aperture, Av, std_z, x0=None, tol=1e-6, max_iter=1000)
        # use FFT operator
        u_hat = np.reshape(u, np.shape(x))
        DMD_u_hat = A_operator(u_hat, aperture)
        DMD_u_hat_v = DMD_u_hat * v
        DMD_u_hat_v_mean += DMD_u_hat_v
    DMD_u_hat_v_mean = DMD_u_hat_v_mean / num_ite_MC_1

    print('gradient 2nd term', 'L=', np.shape(y_mul)[0])
    DMD_h_hat_g_mean = np.zeros_like(x)
    for look_idx in range(np.shape(y_mul)[0]):
        y = y_mul[look_idx]
        # conjugate gd
        h = conjugate_gradient(x, aperture, y, std_z, x0=None, tol=1e-6, max_iter=1000)
        # use FFT operator
        h_hat = np.reshape(h, np.shape(x))
        DMD_h_hat = A_operator(h_hat, aperture)
        DMD_h_hat_g_mean += np.square(DMD_h_hat)

    DMD_h_hat_g_mean /= np.shape(y_mul)[0]
    grad_matrix = 2 * x * (DMD_u_hat_v_mean - DMD_h_hat_g_mean)

    return grad_matrix

# Forward operator implementation of A
def A_operator(h, mask_matrix):
    # consider B = AX^2A^T + sigma^2 I
    D_h = dctn(h, norm="ortho")
    MD_h = D_h * mask_matrix
    DMD_h = idctn(MD_h, norm="ortho")
    return DMD_h

# Efficient operator implementation of AX^2A^T + I
def B_operator(h, x, aperture, std_z):
    # consider B = A X^2 A^H + sigma^2 I
    DMD_h = A_operator(h, aperture)
    x2_DMD_h = np.power(x,2) * DMD_h
    DMD_x2_DMD_h = A_operator(x2_DMD_h, aperture)
    DMD_x2_DMD_h += std_z**2 * h # consider additive term
    return DMD_x2_DMD_h

# Conjugate Gradient Algorithm to solve Ax = b
def conjugate_gradient(x, mask_matrix, b, std_z, x0=None, tol=1e-6, max_iter=1000):
    # Initialization
    if x0 is None:
        u = np.zeros_like(np.reshape(b, (-1,)))  # Initial guess (x0)
    else:
        u = x0.clone()
    r = b - B_operator(np.reshape(u, np.shape(b)), x, mask_matrix, std_z) # Initial residual
    r = np.reshape(r, (-1,))
    p = r.copy()  # Initial direction
    rs_old = np.dot(r, r)  # Inner product of residual

    for i in range(max_iter):
        Ap = np.reshape(B_operator(np.reshape(p, np.shape(b)), x, mask_matrix, std_z), (-1,))
        alpha = rs_old / np.dot(p, Ap)  # Step size
        u = u + alpha * p  # Update estimate of solution
        r = r - alpha * Ap  # Update residual
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            # print('total used ite', i+1)# Check for convergence
            break
        p = r + (rs_new / rs_old) * p  # Update direction
        rs_old = rs_new  # Update residual sum of squares
    return u

