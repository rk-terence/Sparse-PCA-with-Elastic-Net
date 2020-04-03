"""
Sparse PCA according to the paper:
Zou H, Hastie T, Tibshirani R. Sparse Principal Component Analysis[J]. Journal of Computational and Graphical Statistics, 2006, 15(2): 265-286.
Author: rkterence@zju.edu.cn
"""
import numpy as np
from sklearn.linear_model import ElasticNet


def sparse_pca(X, lambda1, lambda2s, n_components, loop_tol=1e-8, input_type='data'):
    """
    Solve Sparse PCA with the form given in paper by Zou H, Hastie T, et al.
    :param X: [ndarray] with shape [n_samples, n_features] The input matrix, which should be centered beforehand.
    :param lambda1: the penalty parameter of ridge regression.
    :param lambda2s: [one dimensional array] the penalty parameters of L1 penalty for each col of B.
    :param n_components: the number of components to extract from X
    :return: The two matrices of established model from SPCA.
        A: [ndarray] with shape [n_features, n_components] the weight matrix
        B: [ndarray] with shape [n_features, n_components] the loading matrix
    """

    # Ordinary PCA
    if input_type == 'data':
        n_samples, n_features = X.shape
        U, S, Vh = np.linalg.svd(X)
        Aold = Vh[:n_components].T  # The initial value of A

        # Compute the square root of X.T @ X
        if n_samples < n_features:
            S = np.hstack((S, np.zeros(n_features-n_samples)))
        Csqrt = (Vh.T * S) @ Vh
    else:  # input_type == 'cov'
        w, v = np.linalg.eig(X)
        idx_sorted = w.argsort()[::-1]
        w = w[idx_sorted]  # sort
        v = v[:, idx_sorted]  # sort
        Aold = v[:, :n_components]
        Csqrt = (v * w**0.5) @ v.T

    Bold = np.zeros_like(Aold)
    n = Csqrt.shape[0]
    # the meta parameters of elastic net
    alphas = np.array([(2 * lambda1 + lambda2s[i]) / (2 * n) for i in range(n_components)])
    l1_ratios = np.array([lambda2s[i] / (2 * lambda1 + lambda2s[i]) for i in range(n_components)])

    iter = 0
    while True:
        # A fixed:
        B = np.zeros_like(Aold)
        for i in range(n_components):
            elastic_net_solver = ElasticNet(alpha=alphas[i], l1_ratio=l1_ratios[i], fit_intercept=False,
                                            max_iter=1e6).fit(Csqrt, Csqrt @ Aold[:, i])
            B[:, i] = elastic_net_solver.coef_

        # B fixed:
        U, S, Vh = np.linalg.svd((X.T @ X) @ B)
        U = U[:, :S.shape[0]]
        Vh = Vh[:S.shape[0]]
        A = U @ Vh

        norm_delta_A = np.linalg.norm(A - Aold, 'fro')
        norm_delta_B = np.linalg.norm(B - Bold, 'fro')
        print("norm(ΔA):", norm_delta_A, "norm(ΔB):", norm_delta_B)
        if norm_delta_A < loop_tol and norm_delta_B < loop_tol:
            break

        iter += 1
        Aold = A
        Bold = B

    B = B / np.linalg.norm(B, 2, axis=0)  # normalization
    return A, B


if __name__ == "__main__":
    # Pipprops data
    Corrcoef = np.array([[0, 0.954, 0.364, 0.342, -0.129, 0.313, 0.496, 0.424, 0.592, 0.545, 0.084, -0.019, 0.134],
                         [0, 0, 0.297, 0.284, -0.118, 0.291, 0.503, 0.419, 0.648, 0.569, 0.076, -0.036, 0.144],
                         [0, 0, 0, 0.882, -0.148, 0.153, -0.029, -0.054, 0.125, -0.081, 0.162, 0.220, 0.126],
                         [0, 0, 0, 0, 0.220, 0.381, 0.174, -0.059, 0.137, -0.014, 0.097, 0.169, 0.015],
                         [0, 0, 0, 0, 0, 0.364, 0.296, 0.004, -0.039, 0.037, -0.091, -0.145, -0.208],
                         [0, 0, 0, 0, 0, 0, 0.813, 0.090, 0.211, 0.274, -0.036, 0.024, -0.329],
                         [0, 0, 0, 0, 0, 0, 0, 0.372, 0.465, 0.679, -0.113, -0.232, -0.424],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0.482, 0.557, 0.061, -0.357, -0.202],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.526, 0.085, -0.127, -0.076],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.319, -0.368, -0.291],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.029, 0.007],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.184],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    Corrcoef = Corrcoef.T + Corrcoef + np.eye(13)
    var_names = ['Topdiam', 'Length', 'Moist', 'Testsg', 'Ovensg', 'Ringtop', 'Ringbut', 'Bowmax', 'Bowdist',
                 'Whorls', 'Clear', 'Knots', 'Diaknot']
    stds = np.array([0.98, 11.29, 57.1, 0.231, 0.070, 3.24, 3.95, 0.43, 9.06, 1.17, 6.27, 1.65, 0.325])
    maximums = np.array([6.00, 60.90, 213.1, 1.217, 0.993, 23, 26, 2.5, 52, 7, 31, 9, 1.65])
    means = np.array([4.21, 46.88, 114.6, 0.877, 0.415, 13.3, 16.3, 0.65, 23.4, 2.49, 10.7, 5.45, 0.82])
    minimums = np.array([2.38, 28.75, 14.7, 0.365, 0.289, 7, 8, 0.13, 7, 1, 1, 0, 0])
    COV = np.array([[Corrcoef[i, j] * stds[i] * stds[j] for j in range(13)] for i in range(13)])

    w, v = np.linalg.eig(Corrcoef)
    lambda1 = 0
    lambda2s = np.array([0.06, 0.16, 0.1, 0.5, 0.5, 0.5])
    A, B = sparse_pca(Corrcoef, lambda1, lambda2s, n_components=6, input_type='cov')
