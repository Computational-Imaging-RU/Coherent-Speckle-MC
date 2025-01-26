import numpy as np
import torch
from scipy.fft import dct, idct, dctn, idctn
from scipy.sparse.linalg import cg
import time

#########################################################
################# Operator + MC ###################
#########################################################

# efficient operator and Monte-Carlo approximation
def nll_grad_operator_MC_CGD(x, y_mul, mask_matrix, std_z, num_ite_MC):
    # first term in gradient

    print('gradient 1st term', 'K=', num_ite_MC)
    num_ite_MC_1 = num_ite_MC
    DMD_u_hat_v_mean = np.zeros_like(x)
    for MC_ite in range(num_ite_MC_1):
        v = np.random.normal(0, 1, size=np.shape(x))
        Av = A_operator(v, mask_matrix)
        # conjugate gd
        u = conjugate_gradient(x, mask_matrix, Av, std_z, x0=None, tol=1e-6, max_iter=1000)
        # use FFT operator
        u_hat = np.reshape(u, np.shape(x))
        D_u_hat = dctn(u_hat, norm="ortho")
        MD_u_hat = D_u_hat * mask_matrix
        DMD_u_hat = idctn(MD_u_hat, norm="ortho")

        DMD_u_hat_v = DMD_u_hat * v
        DMD_u_hat_v_mean += DMD_u_hat_v
    DMD_u_hat_v_mean = DMD_u_hat_v_mean / num_ite_MC_1

    print('gradient 2nd term', 'L=', np.shape(y_mul)[0])
    DMD_h_hat_g_mean = np.zeros_like(x)
    for look_idx in range(np.shape(y_mul)[0]):
        y = y_mul[look_idx]
        # conjugate gd
        h = conjugate_gradient(x, mask_matrix, y, std_z, x0=None, tol=1e-6, max_iter=1000)
        # use FFT operator
        h_hat = np.reshape(h, np.shape(x))
        D_h_hat = dctn(h_hat, norm="ortho")
        MD_h_hat = D_h_hat * mask_matrix
        DMD_h_hat = idctn(MD_h_hat, norm="ortho")
        DMD_h_hat_g_mean += np.square(DMD_h_hat)

    DMD_h_hat_g_mean /= np.shape(y_mul)[0]
    grad_matrix = 2 * x * (DMD_u_hat_v_mean - DMD_h_hat_g_mean)

    return grad_matrix

# efficient operator implementation of AX^2A^T + I
def B_operator(h, x, mask_matrix, std_z):
    # consider B = AX^2A^T + sigma^2 I
    D_h = dctn(h, norm="ortho")
    MD_h = D_h * mask_matrix
    DMD_h = idctn(MD_h, norm="ortho")
    x2_DMD_h = np.power(x,2) * DMD_h
    D_x2_DMD_h = dctn(x2_DMD_h, norm="ortho")
    MD_x2_DMD_h = D_x2_DMD_h * mask_matrix
    DMD_x2_DMD_h = idctn(MD_x2_DMD_h, norm="ortho")
    DMD_x2_DMD_h += std_z**2 * h # consider additive term
    return DMD_x2_DMD_h

def A_operator(h, mask_matrix):
    # consider B = AX^2A^T + sigma^2 I
    D_h = dctn(h, norm="ortho")
    MD_h = D_h * mask_matrix
    DMD_h = idctn(MD_h, norm="ortho")
    return DMD_h

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



#########################################################
################# MatMul + MC ###################
#########################################################


def nll_grad_vector_MC_CGD(x, y_mul, A, mask_matrix, I_n2, std_z, num_ite_MC, dtype, device):

    x_vec = np.reshape(x, (-1,))
    X = np.diag(x_vec)
    X2 = np.square(X)
    x2 = np.square(x)

    B_start = time.time()
    # AX2 = A * np.reshape(x2, (1,-1)) # Broadcasting for column-wise scaling
    # B_ = np.dot(AX2, A)
    # B = B_ + std_z**2 * np.eye(y_mul[0].shape[0]**2) # this is slow
    # B = A @ X2 @ A + std_z ** 2 * np.eye(y_mul[0].shape[0] ** 2)

    A_tensor = torch.from_numpy(A).type(dtype).to(device)
    X2_tensor = torch.from_numpy(X2).type(dtype).to(device)
    B = A_tensor @ X2_tensor @ A_tensor + I_n2
    B = B.detach().cpu().numpy()

    B_end = time.time()
    print('B time', B_end - B_start)

    # first term in gradient
    num_ite_MC_1 = num_ite_MC
    DMD_u_hat_v_mean = np.zeros_like(x)
    for MC_ite in range(num_ite_MC_1):
        v = np.random.normal(0, 1, size=np.shape(x))
        Av = A @ np.reshape(v, (-1,))
        u, _ = cg(B, Av, atol=1e-5)

        # use FFT operator
        u_hat = np.reshape(u, np.shape(x))
        D_u_hat = dctn(u_hat, norm="ortho")
        MD_u_hat = D_u_hat * mask_matrix
        DMD_u_hat = idctn(MD_u_hat, norm="ortho")
        # Matrix multiplication
        # DMD_u_hat = np.reshape(A.T @ u, np.shape(x))

        DMD_u_hat_v = DMD_u_hat * v
        DMD_u_hat_v_mean += DMD_u_hat_v
    DMD_u_hat_v_mean = DMD_u_hat_v_mean / num_ite_MC_1

    DMD_h_hat_g_mean = np.zeros_like(x)
    for look_idx in range(np.shape(y_mul)[0]):
        print('gradient look idx', look_idx)
        y = y_mul[look_idx]
        h, _ = cg(B, np.reshape(y, (-1,)), atol=1e-5)

        # use FFT operator
        h_hat = np.reshape(h, np.shape(x))
        D_h_hat = dctn(h_hat, norm="ortho")
        MD_h_hat = D_h_hat * mask_matrix
        DMD_h_hat = idctn(MD_h_hat, norm="ortho")
        DMD_h_hat_g_mean += np.square(DMD_h_hat)
        # Matrix multiplication
        # DMD_h_hat_g_mean += np.square(np.reshape(A @ h, np.shape(x)))

    DMD_h_hat_g_mean /= np.shape(y_mul)[0]

    grad_matrix = 2 * x * (DMD_u_hat_v_mean - DMD_h_hat_g_mean)

    return grad_matrix


def nll_grad_vector(x, y_mul, A, I_n2, std_z, dtype, device):
    y = y_mul[0]
    x_vec = np.reshape(x, (-1,))
    X = np.diag(x_vec)
    X2 = np.square(X)

    B_start = time.time()
    B = A @ X2 @ A + std_z ** 2 * np.eye(y.shape[0] ** 2)
    B_end = time.time()
    print('B time', B_end - B_start)
    B_inv_start = time.time()
    B_inv = np.linalg.inv(B)
    B_inv_end = time.time()
    print('B inv time', B_inv_end - B_inv_start)
    g1_start = time.time()
    DMD_u_hat_v_mean = np.reshape(np.diagonal(A.T @ B_inv @ A), np.shape(x))
    g1_end = time.time()
    g2_start = time.time()
    DMD_h_hat_g_mean = np.reshape(A @ B_inv @ np.reshape(y, (-1,)), np.shape(x))
    g2_end = time.time()
    print('g1 time', g1_end - g1_start, 'g2 time', g2_end - g2_start)

    # A_tensor = torch.from_numpy(A).type(dtype).to(device)
    # X2_tensor = torch.from_numpy(X2).type(dtype).to(device)
    # B_start = time.time()
    # B = A_tensor @ X2_tensor @ A_tensor + I_n2
    # B_end = time.time()
    # print('B time', B_end - B_start)
    # B_inv_start = time.time()
    # B_inv = torch.linalg.inv(B)
    # B_inv_end = time.time()
    # print('B inv time', B_inv_end - B_inv_start)
    # g1_start = time.time()
    # DMD_u_hat_v_mean = torch.reshape(torch.diag(A_tensor.T @ B_inv @ A_tensor), x.shape).detach().cpu().numpy()
    # g1_end = time.time()
    # g2_start = time.time()
    # DMD_h_hat_g_mean = torch.reshape(A_tensor @ B_inv @ torch.from_numpy(np.reshape(y, (-1,))).type(dtype).to(device), x.shape).detach().cpu().numpy()
    # g2_end = time.time()
    # print('g1 time', g1_end - g1_start, 'g2 time', g2_end - g2_start)

    grad_matrix = 2 * x * (DMD_u_hat_v_mean - np.square(DMD_h_hat_g_mean))

    return grad_matrix




def nll_grad_vector_MC_approximation(x, y_mul, A, mask_matrix, std_z):
    y = y_mul[0]

    x_vec = np.reshape(x, (-1,))
    X = np.diag(x_vec)
    X2 = np.square(X)
    # B = A @ X2 @ A
    B = A @ X2 @ A + std_z**2 * np.eye(y.shape[0]**2)
    # B_inv = np.linalg.inv(B)

    num_ite_MC_2 = 5
    num_ite_MC_1 = 5
    # num_ite_optimization = 1000000

    # second term in gradient
    # h_lr = 1.0
    # MC approximation
    DMD_h_hat_g_mean = np.zeros_like(x)
    for MC_ite in range(num_ite_MC_2):
        # print('MC ite', MC_ite)
        g = np.random.normal(0, 1, size=np.shape(x))

        # B_inv_y = B_inv @ np.reshape(y, (-1,))
        # B_inv_y = B_inv @ (np.reshape(y, (-1,)) * np.reshape(g, (-1,)))
        # B_inv_y_ref = np.reshape(B_inv_y, (y.shape[0], y.shape[1]))
        # h = B_inv_y_ref
        # h = np.ones_like(x)
        # h = np.random.normal(0, 1, size=np.shape(x))

        yg = np.reshape(np.multiply(y, g), (-1,))
        h, _ = cg(B, yg, atol=1e-5)
        # print('MSE in cgd', np.mean(np.square(h - B_inv_y)))

        # optimization over u
        # for h_ite in range(num_ite_optimization):
            # only consider B = AX^2A^T
            # D_h = dctn(h, norm="ortho")
            # MD_h = D_h * mask_matrix
            # DMD_h = idctn(MD_h, norm="ortho")
            # x2_DMD_h = np.power(x,2) * DMD_h
            # D_x2_DMD_h = dctn(x2_DMD_h, norm="ortho")
            # MD_x2_DMD_h = D_x2_DMD_h * mask_matrix
            # DMD_x2_DMD_h = idctn(MD_x2_DMD_h, norm="ortho")
            # D_DMD_x2_DMD_h = dctn(DMD_x2_DMD_h, norm="ortho")
            # MD_DMD_x2_DMD_h = D_DMD_x2_DMD_h * mask_matrix
            # DMD_DMD_x2_DMD_h = idctn(MD_DMD_x2_DMD_h, norm="ortho")
            # x2_DMD_DMD_x2_DMD_h = np.power(x,2) * DMD_DMD_x2_DMD_h
            # D_x2_DMD_DMD_x2_DMD_h = dctn(x2_DMD_DMD_x2_DMD_h, norm="ortho")
            # MD_x2_DMD_DMD_x2_DMD_h = D_x2_DMD_DMD_x2_DMD_h * mask_matrix
            # DMD_x2_DMD_DMD_x2_DMD_h = idctn(MD_x2_DMD_DMD_x2_DMD_h, norm="ortho")
            #
            # yg = np.multiply(y, g)
            # D_yg = dctn(yg, norm="ortho")
            # MD_yg = D_yg * mask_matrix
            # DMD_yg = idctn(MD_yg, norm="ortho")
            # x2_DMD_yg = x**2 * DMD_yg
            # D_x2_DMD_yg = dctn(x2_DMD_yg, norm="ortho")
            # MD_x2_DMD_yg = D_x2_DMD_yg * mask_matrix
            # DMD_x2_DMD_yg = idctn(MD_x2_DMD_yg, norm="ortho")
            # h_grad = 2 * (DMD_x2_DMD_yg - DMD_x2_DMD_DMD_x2_DMD_h)
            # h += h_lr * h_grad
            # # print(h_grad)
            # print('ite', h_ite, 'MSE in optimization', np.mean(np.square(h - B_inv_y_ref)))

            # consider B = AX^2A^T + sigma^2 I
            # D_h = dctn(h, norm="ortho")
            # MD_h = D_h * mask_matrix
            # DMD_h = idctn(MD_h, norm="ortho")
            # x2_DMD_h = np.power(x,2) * DMD_h
            # D_x2_DMD_h = dctn(x2_DMD_h, norm="ortho")
            # MD_x2_DMD_h = D_x2_DMD_h * mask_matrix
            # DMD_x2_DMD_h = idctn(MD_x2_DMD_h, norm="ortho")
            # DMD_x2_DMD_h += 0.04**2 * h # consider additive term
            #
            # D_DMD_x2_DMD_h = dctn(DMD_x2_DMD_h, norm="ortho")
            # MD_DMD_x2_DMD_h = D_DMD_x2_DMD_h * mask_matrix
            # DMD_DMD_x2_DMD_h = idctn(MD_DMD_x2_DMD_h, norm="ortho")
            # x2_DMD_DMD_x2_DMD_h = np.power(x,2) * DMD_DMD_x2_DMD_h
            # D_x2_DMD_DMD_x2_DMD_h = dctn(x2_DMD_DMD_x2_DMD_h, norm="ortho")
            # MD_x2_DMD_DMD_x2_DMD_h = D_x2_DMD_DMD_x2_DMD_h * mask_matrix
            # DMD_x2_DMD_DMD_x2_DMD_h = idctn(MD_x2_DMD_DMD_x2_DMD_h, norm="ortho")
            # DMD_x2_DMD_DMD_x2_DMD_h += 0.04**2 * D_DMD_x2_DMD_h # consider additive term
            #
            # yg = np.multiply(y, g)
            # D_yg = dctn(yg, norm="ortho")
            # MD_yg = D_yg * mask_matrix
            # DMD_yg = idctn(MD_yg, norm="ortho")
            # x2_DMD_yg = x**2 * DMD_yg
            # D_x2_DMD_yg = dctn(x2_DMD_yg, norm="ortho")
            # MD_x2_DMD_yg = D_x2_DMD_yg * mask_matrix
            # DMD_x2_DMD_yg = idctn(MD_x2_DMD_yg, norm="ortho")
            # DMD_x2_DMD_yg += 0.04**2 * yg # consider additive term
            #
            # h_grad = 2 * (DMD_x2_DMD_yg - DMD_x2_DMD_DMD_x2_DMD_h)
            # h += h_lr * h_grad
            # # print(h_grad)
            # print('ite', h_ite, 'MSE in optimization', np.mean(np.square(h - B_inv_y_ref)))

        h_hat = np.reshape(h, np.shape(x))
        D_h_hat = dctn(h_hat, norm="ortho")
        MD_h_hat = D_h_hat * mask_matrix
        DMD_h_hat = idctn(MD_h_hat, norm="ortho")
        DMD_h_hat_g = DMD_h_hat * g
        # DMD_h_hat_g = DMD_h_hat
        DMD_h_hat_g_mean += DMD_h_hat_g

        # DMD_h_hat_g_mean = A.T @ B_inv_y
        # DMD_h_hat_g_mean = np.reshape(DMD_h_hat_g_mean, np.shape(x))

    DMD_h_hat_g_mean = DMD_h_hat_g_mean / num_ite_MC_2


    # first term in gradient
    # u_lr = 0.1
    # MC approximation
    DMD_u_hat_v_mean = np.zeros_like(x)
    for MC_ite in range(num_ite_MC_1):
        # print('MC ite', MC_ite)
        v = np.random.normal(0, 1, size=np.shape(x))

        # AT_B_inv_A = A.T @ B_inv @ A
        # DMD_u_hat_v_mean_ = np.diagonal(AT_B_inv_A)
        # DMD_u_hat_v_mean = np.reshape(DMD_u_hat_v_mean_, np.shape(x))

        # B_inv_Av = B_inv @ A @ np.reshape(v, (-1,))
        # B_inv_Av_ref = np.reshape(B_inv_Av, (y.shape[0], y.shape[1]))
        # u = B_inv_Av_ref

        Av = A @ np.reshape(v, (-1,))
        u, _ = cg(B, Av, atol=1e-5)
        # u = np.ones_like(x)

        # optimization over u
        # for u_ite in range(num_ite_optimization):
        #     D_u = dctn(u, norm="ortho")
        #     MD_u = D_u * mask_matrix
        #     DMD_u = idctn(MD_u, norm="ortho")
        #     x2_DMD_u = np.power(x,2) * DMD_u
        #     D_x2_DMD_u = dctn(x2_DMD_u, norm="ortho")
        #     M2D_x2_DMD_u = D_x2_DMD_u * mask_matrix**2
        #     DM2D_x2_DMD_u = idctn(M2D_x2_DMD_u, norm="ortho")
        #     x2_DM2D_x2_DMD_u = np.power(x,2) * DM2D_x2_DMD_u
        #     D_x2_DM2D_x2_DMD_u = dctn(x2_DM2D_x2_DMD_u, norm="ortho")
        #     MD_x2_DM2D_x2_DMD_u = D_x2_DM2D_x2_DMD_u * mask_matrix
        #     DMD_x2_DM2D_x2_DMD_u = idctn(MD_x2_DM2D_x2_DMD_u, norm="ortho")
        #
        #     D_v = dctn(v, norm="ortho")
        #     M2D_v = D_v * mask_matrix**2
        #     DM2D_v = idctn(M2D_v, norm="ortho")
        #     x2_DM2D_v = np.multiply(x ** 2, DM2D_v)
        #     D_x2_DM2D_v = dctn(x2_DM2D_v, norm="ortho")
        #     MD_x2_DM2D_v = D_x2_DM2D_v * mask_matrix
        #     DMD_x2_DM2D_v = idctn(MD_x2_DM2D_v, norm="ortho")
        #     u += u_lr * (DMD_x2_DM2D_v - DMD_x2_DM2D_x2_DMD_u)

        u_hat = np.reshape(u, np.shape(x))
        D_u_hat = dctn(u_hat, norm="ortho")
        MD_u_hat = D_u_hat * mask_matrix
        DMD_u_hat = idctn(MD_u_hat, norm="ortho")
        DMD_u_hat_v = DMD_u_hat * v
        DMD_u_hat_v_mean += DMD_u_hat_v
    DMD_u_hat_v_mean = DMD_u_hat_v_mean / num_ite_MC_1

    grad_matrix = 2 * x * (DMD_u_hat_v_mean - np.square(DMD_h_hat_g_mean))

    return grad_matrix




################################################
################ Newton Schulz ################
################################################



def nll_grad_vector_blur_additive(x, A, y, std_w, std_z, patch_size, device, dtype):
    X = torch.diag_embed(x)  # Create diagonal matrices from x
    X2 = torch.square(X)

    B = std_w**2 * A @ X2 @ A + std_z**2 * torch.eye(patch_size**2).type(dtype).to(device)

    B_inv_current = torch.inverse(B)

    ATB_inv = torch.matmul(A, B_inv_current)
    grad_vec_1 = torch.diagonal(torch.matmul(ATB_inv, A), dim1=-2, dim2=-1)
    grad_vec_2_ = torch.square(torch.matmul(ATB_inv, y))
    grad_vec_2 = torch.mean(grad_vec_2_, dim=-1)  # average over all measurement looks
    assert grad_vec_1.shape == grad_vec_2.shape == x.shape
    grad_vec = 2 * x * (grad_vec_1 * std_w ** 2 - grad_vec_2)

    return grad_vec, B_inv_current

def nll_grad_vector_blur_additive_approx(x, A, y, B_inv, std_w, std_z, patch_size, device, dtype, use_Schulz):
    X = torch.diag_embed(x)  # Create diagonal matrices from x
    X2 = torch.square(X)

    B = std_w**2 * A @ X2 @ A + std_z**2 * torch.eye(patch_size**2).type(dtype).to(device)
    if use_Schulz:
        print('Approx matrix inverse with Newton-Schulz')
        for Newton_i in range(1):
            B_inv = 2 * B_inv - B_inv @ B @ B_inv
        B_inv_current = B_inv
    else:
        B_inv_current = torch.inverse(B)

    ATB_inv = torch.matmul(A, B_inv_current)
    grad_vec_1 = torch.diagonal(torch.matmul(ATB_inv, A), dim1=-2, dim2=-1)
    grad_vec_2_ = torch.square(torch.matmul(ATB_inv, y))
    grad_vec_2 = torch.mean(grad_vec_2_, dim=-1)  # average over all measurement looks
    assert grad_vec_1.shape == grad_vec_2.shape == x.shape
    grad_vec = 2 * x * (grad_vec_1 * std_w ** 2 - grad_vec_2)

    return grad_vec, B_inv_current

