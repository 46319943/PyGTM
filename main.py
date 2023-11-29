import json

import numpy as np
from numpy.linalg import det
from numba import njit, prange
from numba import types, typed

from scipy.optimize import minimize

from gensim.corpora import Dictionary
from scipy.spatial.distance import pdist, squareform

np.random.seed(10)


@njit()
def entropy_phi(phis):
    D = len(phis)
    term = 0
    for document_index in range(D):
        phi = phis[document_index]
        term += np.sum(phi * np.log(phi))
    return -term


@njit()
def entropy_nu2(nu2s):
    return 0.5 * np.sum(np.log(nu2s) + np.log(2 * np.pi) + 1)


@njit()
def entropy_psi2(psi2s):
    return 0.5 * np.sum(np.log(psi2s) + np.log(2 * np.pi) + 1)


@njit()
def ELBO_lam(sigma_invs, phis, zetas, lams, nu2s, omegas, locations):
    D = len(phis)

    log_p_eta = 0
    log_p_z = 0
    for document_index in range(D):
        sigma_inv = sigma_invs[locations[document_index]]
        lam = lams[document_index]
        omega = omegas[locations[document_index]]
        log_p_eta += -0.5 * np.dot(np.dot((lam - omega).T, sigma_inv), lam - omega)

        zeta = zetas[document_index]
        nu2 = nu2s[document_index]
        word_count = len(phis[document_index])
        term1 = np.sum(np.dot(lam, phis[document_index].T))
        term2 = - word_count * (1 / zeta) * sum(np.exp(lam + nu2 / 2))
        log_p_z += term1 + term2

    return log_p_eta + log_p_z


@njit()
def df_lam(sigma_invs, phis, zetas, lams, nu2s, omegas, locations):
    D = len(phis)
    T = lams.shape[1]

    term1 = np.zeros((D, T))
    word_counts = np.zeros(D)
    for document_index in range(D):
        term1[document_index] = np.sum(phis[document_index], axis=0)
        word_counts[document_index] = len(phis[document_index])

    term2 = - (word_counts / zetas)[:, np.newaxis] * np.exp(lams + nu2s / 2)

    term3 = np.zeros((D, T))
    for document_index in range(D):
        sigma_inv = sigma_invs[locations[document_index]]
        lam = lams[document_index]
        omega = omegas[locations[document_index]]
        term3[document_index] = - np.dot(sigma_inv, lam - omega)

    return term1 + term2 + term3


@njit(parallel=True, fastmath=True)
def numba_logsumexp_stable(p, out):
    n, m = p.shape
    assert len(out) == n
    assert out.ndim == 1
    assert p.ndim == 2

    for i in prange(n):
        p_max = np.max(p[i])
        res = 0
        for j in range(m):
            res += np.exp(p[i, j] - p_max)
        res = np.log(res) + p_max
        out[i] = res

@njit()
def opt_phi(log_beta, lams, corpus):
    D = len(lams)
    T = lams.shape[1]

    phis = typed.List.empty_list(types.float64[:, ::1])
    for document_index in range(D):
        lam = lams[document_index]
        N = len(corpus[document_index])
        log_phi = np.zeros((N, T), dtype=np.float64)

        for n, word in enumerate(corpus[document_index]):
            log_phi[n, :] = lam + log_beta[:, word]

        # This code segment about log-sum-exp is optimized using GPT4.
        # Using NumPy operations for log-sum-exp
        # max_log_phi = np.max(log_phi, axis=1, keepdims=True)
        # log_phi_sum = np.log(np.sum(np.exp(log_phi - max_log_phi), axis=1, keepdims=True)) + max_log_phi

        # Using Numba operations for log-sum-exp
        log_phi_sum = np.zeros((N, ), dtype=np.float64)
        numba_logsumexp_stable(log_phi, log_phi_sum)
        log_phi_sum = log_phi_sum[:, np.newaxis]

        # Vectorized calculation of phi
        phi = np.exp(log_phi - log_phi_sum)

        phis.append(phi)

    return phis

@njit()
def log_p_w(log_beta, phis: types.ListType(types.float64[:, ::1]), corpus):
    D = len(phis)
    value = 0
    for document_index in range(D):
        phi = phis[document_index]
        for n, word in enumerate(corpus[document_index]):
            value += np.dot(phi[n], log_beta[:, word])
    return value

@njit()
def log_p_z(phis, zetas, lams, nu2s):
    D = len(phis)
    value = 0
    for document_index in range(D):
        zeta = zetas[document_index]
        nu2 = nu2s[document_index]
        lam = lams[document_index]
        phi = phis[document_index]

        word_count = len(phi)

        term1 = np.sum(np.dot(lam, phi.T))
        term2 = - word_count * (1 / zeta) * sum(np.exp(lam + nu2 / 2))
        term3 = word_count * (1 - np.log(zeta))
        value += term1 + term2 + term3
    return value

@njit()
def log_p_eta(sigma_invs, lams, nu2s, psi2s, omegas, locations):
    D = len(lams)
    T = lams.shape[1]

    value = 0
    for document_index in range(D):
        sigma_inv = sigma_invs[locations[document_index]]
        lam = lams[document_index]
        omega = omegas[locations[document_index]]
        nu2 = nu2s[document_index]
        psi2 = psi2s[locations[document_index]]

        term1 = 0.5 * np.log(det(sigma_inv))
        term2 = 0.5 * T * np.log(2 * np.pi)
        term3 = 0.5 * (
                np.trace(np.dot(np.diag(nu2), sigma_inv)) +
                np.trace(np.dot(np.diag(psi2), sigma_inv)) +
                np.dot(np.dot((lam - omega).T, sigma_inv), lam - omega)
        )

        value += term1 - term2 - term3

    return value

@njit()
def log_p_mu(weight_matrix_inv, weight_matrix_inv_det, omegas, psi2s, m):
    L = omegas.shape[0]
    T = omegas.shape[1]

    value = 0
    for topic_index in range(T):
        psi2 = psi2s[:, topic_index]
        omega = omegas[:, topic_index]
        omega_minus_m = omega - m

        term1 = 0.5 * np.log(weight_matrix_inv_det)
        term2 = 0.5 * L * np.log(2 * np.pi)
        term3 = 0.5 * (
                np.trace(np.dot(np.diag(psi2), weight_matrix_inv)) +
                np.dot(np.dot(omega_minus_m.T, weight_matrix_inv), omega_minus_m)
        )

        value += term1 - term2 - term3

    return value

# @njit()
def ELBO(
        log_beta, m, sigma_invs, weight_matrix_inv, weight_matrix_inv_det,
        phis, zetas, lams, nu2s, omegas, psi2s, locations, corpus):
    return (
            log_p_w(log_beta, phis, corpus) +
            log_p_z(phis, zetas, lams, nu2s) +
            log_p_eta(sigma_invs, lams, nu2s, psi2s, omegas, locations) +
            log_p_mu(weight_matrix_inv, weight_matrix_inv_det, omegas, psi2s, m) +
            entropy_phi(phis) + entropy_nu2(nu2s) + entropy_psi2(psi2s)
    )


class GTM:
    def __init__(self, topic_count, vocab_size, location_count, weight_matrix):
        # Model Property
        self.topic_count = topic_count
        self.vocab_size = vocab_size
        self.location_count = location_count

        # Model parameter
        self.weight_matrix = weight_matrix
        self.weight_matrix_inv = np.empty((0, 0), dtype=np.float64)
        self.weight_matrix_inv_det = np.float64(0)
        self.m = np.empty((0,), dtype=np.float64)
        self.sigma = np.empty((0, 0, 0), dtype=np.float64)
        self.sigma_inv = np.empty((0, 0, 0), dtype=np.float64)
        self.sigma_inv_det = np.empty((0,), dtype=np.float64)
        self.beta = np.empty((0, 0), dtype=np.float64)
        self.log_beta = np.empty((0, 0), dtype=np.float64)

        # Training parameters
        self.corpus = typed.List.empty_list(types.int32[::1])
        self.locations = np.empty((0,), dtype=np.int32)
        self.document_size = 0
        self.word_counts = np.empty((0,), dtype=np.int32)

        # Variational factors
        self.phi = typed.List.empty_list(types.float64[:, ::1])
        self.zeta = np.empty((0,), dtype=np.float64)
        self.lam = np.empty((0, 0), dtype=np.float64)
        self.nu2 = np.empty((0, 0), dtype=np.float64)
        self.omega = np.empty((0, 0), dtype=np.float64)
        self.psi2 = np.empty((0, 0), dtype=np.float64)

        self.init_model_param()

    def init_model_param(self):
        # Size(location_count)
        self.m = np.zeros(self.location_count, dtype=np.float64)
        # Size(location_count, topic_count, topic_count)
        # self.sigma = np.eye(self.topic_count, dtype=np.float64).repeat(self.location_count).reshape(
        #     (self.location_count, self.topic_count, self.topic_count))
        self.sigma = np.tile(np.eye(self.topic_count, dtype=np.float64), (self.location_count, 1, 1))
        self.sigma_inv = self.sigma
        self.sigma_inv_det = np.ones(self.location_count, dtype=np.float64)
        # Size(topic_count, vocab_size)
        self.beta = np.float64(0.001) + np.random.uniform(0, 1, (self.topic_count, self.vocab_size)).astype(np.float64)

        # for i in range(self.topic_count):
        #     beta[i] /= sum(beta[i])
        self.beta /= np.sum(self.beta, axis=1)[:, np.newaxis]

        self.log_beta = np.log(self.beta)

        self.weight_matrix_inv = np.linalg.inv(self.weight_matrix)
        self.weight_matrix_inv_det = np.linalg.det(self.weight_matrix_inv)

    def init_variational_factor(self):
        self.zeta = np.ones(self.document_size, dtype=np.float64) * 10

        self.lam = np.zeros((self.document_size, self.topic_count), dtype=np.float64)
        self.nu2 = np.ones((self.document_size, self.topic_count), dtype=np.float64)

        self.omega = np.zeros((self.location_count, self.topic_count), dtype=np.float64)
        self.psi2 = np.ones((self.location_count, self.topic_count), dtype=np.float64)

        # The size of phi is dependent on the words count in each document
        phi_init_value = np.float64(1 / self.topic_count)
        for i in range(self.document_size):
            self.phi.append(
                np.ones((self.word_counts[i], self.topic_count), dtype=np.float64) * phi_init_value
            )

    def train(self, corpus, locations, max_iter=1000):
        self.corpus = corpus
        self.locations = locations
        self.document_size = len(corpus)
        self.word_counts = np.array([len(doc) for doc in corpus])
        self.init_variational_factor()

        after = ELBO(
            self.log_beta, self.m, self.sigma_inv, self.weight_matrix_inv, self.weight_matrix_inv_det,
            self.phi, self.zeta, self.lam, self.nu2, self.omega, self.psi2, self.locations, self.corpus
        )

        for _ in range(max_iter):
            before = after
            self.expectation()
            self.maximization()

            after = ELBO(
                self.log_beta, self.m, self.sigma_inv, self.weight_matrix_inv, self.weight_matrix_inv_det,
                self.phi, self.zeta, self.lam, self.nu2, self.omega, self.psi2, self.locations, self.corpus
            )

            print('lhood = ', after)
            print(((before - after) / before))
            if ((before - after) / before) < 0.001:
                break

    # def lhood_bnd_involved_omega(self, omega_v):
    #     omega_v = omega_v.reshape((self.location_count, self.topic_count))
    #     omegas = omega_v
    #
    #     log_p_doc = 0
    #
    #     for document_index in range(self.document_size):
    #         location_index = self.locations[document_index]
    #         omega = omegas[location_index]
    #         lam = self.lam[document_index]
    #
    #         sigma_inv = self.sigma_inv[location_index]
    #         lam_minus_omega = lam - omega
    #
    #         term3 = 0.5 * np.dot(np.dot(lam_minus_omega.T, sigma_inv), lam_minus_omega)
    #
    #         log_p_doc += -term3
    #
    #     log_p_topic = 0
    #
    #     for topic_index in range(self.topic_count):
    #         omega = omegas[:, topic_index]
    #         weight_matrix_inv = self.weight_matrix_inv
    #         m = self.m
    #
    #         omega_minus_m = omega - m
    #
    #         term3 = 0.5 * np.dot(np.dot(omega_minus_m.T, weight_matrix_inv), omega_minus_m)
    #
    #         log_p_topic += -term3
    #
    #     return log_p_doc + log_p_topic

    def lhood_bnd_involved_omega(self, omega_v):
        omega_v = omega_v.reshape((self.location_count, self.topic_count))

        # Vectorized computation for the document loop
        location_indices = self.locations[:self.document_size]
        lam = self.lam[:self.document_size]
        omega = omega_v[location_indices]
        sigma_inv = self.sigma_inv[location_indices]
        lam_minus_omega = lam - omega
        # term3_docs = 0.5 * np.einsum('ij,ij->i', lam_minus_omega, np.dot(sigma_inv, lam_minus_omega))
        term3_docs = 0.5 * np.einsum('ijk,ik->i', sigma_inv, lam_minus_omega)
        log_p_doc = -np.sum(term3_docs)

        # Vectorized computation for the topic loop
        omega_minus_m = omega_v - self.m[:, np.newaxis]
        term3_topics = 0.5 * np.einsum('ij,ij->j', omega_minus_m, np.dot(self.weight_matrix_inv, omega_minus_m))
        log_p_topic = -np.sum(term3_topics)

        return log_p_doc + log_p_topic

    def opt_lam(self):
        fn = lambda x: - ELBO_lam(self.sigma_inv, self.phi, self.zeta,
                                  x.reshape((self.document_size, self.topic_count)), self.nu2, self.omega,
                                  self.locations)
        g = lambda x: - df_lam(self.sigma_inv, self.phi, self.zeta, x.reshape((self.document_size, self.topic_count)),
                               self.nu2, self.omega, self.locations).flatten()

        res = minimize(fn, x0=self.lam.flatten(), jac=g, method='Newton-CG', options={'disp': 0})
        lam_optimized = res.x

        self.lam = lam_optimized.reshape((self.document_size, self.topic_count))

    # def df_nu2(self, nu2):
    #     term1 = 0.5 * np.diag(self.sigma_inv[self.locations])
    #     term2 = 0.5 * (self.word_counts / self.zeta) * np.exp(self.lam + nu2 / 2)
    #     term3 = 1 / (2 * nu2)
    #     return -term1 - term2 + term3
    #
    # def df2_nu2(self, nu2):
    #     term1 = - 0.25 * (self.word_counts / self.zeta) * np.exp(self.lam + nu2 / 2)
    #     term2 = - 0.5 * (1 / (nu2 * nu2))
    #     return term1 + term2
    #
    # def opt_nu2(self, document_index):
    #     g = lambda nu2: self.df_nu2(nu2)
    #     h = lambda nu2: self.df2_nu2(nu2)
    #
    #     init_x = np.ones(self.topic_count) * 10
    #     x = init_x
    #
    #     log_x = np.log(x)
    #     df1 = np.ones(self.topic_count)
    #
    #     while np.all(np.abs(df1) > 0.0001):
    #         if np.any(np.isnan(x)):
    #             init_x = init_x * 10
    #             x = init_x
    #             log_x = np.log(x)
    #         x = np.exp(log_x)
    #
    #         df1 = g(x)
    #         df2 = h(x)
    #
    #         log_x -= (x * df1) / (x * x * df2 + x * df1)
    #
    #     self.nu2[document_index] = np.exp(log_x)

    def opt_nu2(self):
        D = self.document_size
        T = self.topic_count

        # Precompute constants and reshape for batch operations
        sigma_inv_diag = 0.5 * np.array([np.diag(self.sigma_inv[self.locations[i]]) for i in range(D)])
        zeta = self.zeta
        lam = self.lam

        # Initialize variables for batch optimization
        init_x = np.ones((D, T)) * 10
        x = init_x
        log_x = np.log(x)

        while True:
            x = np.exp(log_x)

            # Batch calculations for df_nu2 and df2_nu2
            term2_nu2 = 0.5 * (self.word_counts / zeta)[:, np.newaxis] * np.exp(lam + x / 2)
            term3_nu2 = 1 / (2 * x)
            df1 = -(sigma_inv_diag + term2_nu2 - term3_nu2)

            term1_nu2_2 = 0.25 * (self.word_counts / zeta)[:, np.newaxis] * np.exp(lam + x / 2)
            term2_nu2_2 = 0.5 * (1 / (x * x))
            df2 = -(term1_nu2_2 + term2_nu2_2)

            log_x -= (x * df1) / (x * x * df2 + x * df1)

            # Check for convergence and NaN handling
            if np.all(np.abs(df1) <= 0.0001):
                break
            if np.any(np.isnan(x)):
                init_x *= 10
                log_x = np.log(init_x)

        self.nu2 = x

    def opt_zeta(self):
        self.zeta = np.sum(np.exp(self.lam + self.nu2 / 2), axis=-1)

    # def df_omega(self, omega_v):
    #     if omega_v.size == 1:
    #         omega_single = omega_v
    #         omega_v = self.omega
    #         omega_v[0, 0] = omega_single
    #
    #     omega_v = omega_v.reshape((self.location_count, self.topic_count))
    #     omega_d = np.zeros((self.location_count, self.topic_count))
    #
    #     for location_index in range(self.location_count):
    #         omega = omega_v[location_index]
    #         sigma_inv = self.sigma_inv[location_index]
    #
    #         term1 = 0
    #         for document_index in np.nonzero(self.locations == location_index)[0]:
    #             lam = self.lam[document_index]
    #             lam_minus_omega = lam - omega
    #             term1 += np.dot(sigma_inv, lam_minus_omega)
    #
    #         omega_d[location_index] += term1
    #
    #     for topic_index in range(self.topic_count):
    #         omega = omega_v[:, topic_index]
    #         weight_matrix_inv = self.weight_matrix_inv
    #         m = self.m
    #
    #         term2 = np.dot(weight_matrix_inv, omega - m)
    #
    #         omega_d[:, topic_index] += -term2
    #
    #     return omega_d

    def df_omega(self, omega_v):
        omega_v = omega_v.reshape((self.location_count, self.topic_count))

        # Vectorized computation for location loop
        omega_d_location = np.zeros((self.location_count, self.topic_count))
        for location_index in range(self.location_count):
            sigma_inv = self.sigma_inv[location_index]
            document_indices = np.nonzero(self.locations == location_index)[0]
            lam = self.lam[document_indices]
            lam_minus_omega = lam - omega_v[location_index]
            omega_d_location[location_index] = np.dot(sigma_inv, lam_minus_omega.sum(axis=0))

        # Vectorized computation for topic loop
        omega_minus_m = omega_v - self.m[:, np.newaxis]
        omega_d_topic = -np.dot(self.weight_matrix_inv, omega_minus_m)

        return omega_d_location + omega_d_topic

    def opt_omega(self):
        omega = self.omega.flatten()

        fn = lambda x: - self.lhood_bnd_involved_omega(x)
        g = lambda x: - self.df_omega(x).flatten()

        res = minimize(fn, x0=omega, jac=g, method='Newton-CG', options={'disp': 0})
        omega_optimized = res.x

        self.omega = omega_optimized.reshape((self.location_count, self.topic_count))

    # def df_psi2(self, psi2):
    #     psi2_d = np.zeros((self.location_count, self.topic_count))
    #
    #     for location_index in range(self.location_count):
    #         sigma_inv = self.sigma_inv[location_index]
    #         term1 = 0.5 * np.diag(sigma_inv)
    #         psi2_d[location_index] += term1
    #
    #     for topic_index in range(self.topic_count):
    #         weight_matrix_inv = self.weight_matrix_inv
    #         term2 = 0.5 * np.diag(weight_matrix_inv)
    #         psi2_d[:, topic_index] += term2
    #
    #     psi2_d = -psi2_d
    #
    #     psi2 = psi2.reshape((self.location_count, self.topic_count))
    #     term3 = 1 / (2 * psi2)
    #     psi2_d += term3
    #
    #     return psi2_d
    #
    # def df2_psi2(self, psi2):
    #     psi2 = psi2.reshape((self.location_count, self.topic_count))
    #     return - 0.5 * (1 / (psi2 * psi2))
    #
    # def opt_psi2(self):
    #     g = lambda psi2: self.df_psi2(psi2).flatten()
    #     h = lambda psi2: self.df2_psi2(psi2).flatten()
    #
    #     init_x = np.ones((self.location_count, self.topic_count)).flatten() * 10
    #     x = init_x
    #
    #     log_x = np.log(x)
    #     df1 = np.ones((self.location_count, self.topic_count)).flatten()
    #
    #     while np.all(np.abs(df1) > 0.0001):
    #         if np.any(np.isnan(x)):
    #             init_x = init_x * 10
    #             x = init_x
    #             log_x = np.log(x)
    #         x = np.exp(log_x)
    #
    #         df1 = g(x)
    #         df2 = h(x)
    #
    #         log_x -= (x * df1) / (x * x * df2 + x * df1)
    #
    #     self.psi2 = np.exp(log_x).reshape((self.location_count, self.topic_count))

    def df_psi2(self, psi2):
        term1 = 0.5 * np.diagonal(self.sigma_inv, axis1=1, axis2=2)
        term2 = 0.5 * np.diag(self.weight_matrix_inv)

        psi2_d = - (term1 + term2[:, np.newaxis])

        psi2 = psi2.reshape((self.location_count, self.topic_count))
        term3 = 1 / (2 * psi2)
        psi2_d += term3

        return psi2_d

    def df2_psi2(self, psi2):
        psi2 = psi2.reshape((self.location_count, self.topic_count))
        return -0.5 * (1 / (psi2 * psi2))

    def opt_psi2(self):
        g = lambda psi2: self.df_psi2(psi2).flatten()
        h = lambda psi2: self.df2_psi2(psi2).flatten()

        init_x = np.ones((self.location_count, self.topic_count)).flatten() * 10
        x = init_x

        log_x = np.log(x)
        df1 = np.ones_like(x)

        while np.all(np.abs(df1) > 0.0001):
            if np.any(np.isnan(x)):
                init_x *= 10
                x = init_x
                log_x = np.log(x)
            x = np.exp(log_x)

            df1 = g(x)
            df2 = h(x)

            log_x -= (x * df1) / (x * x * df2 + x * df1)

        self.psi2 = np.exp(log_x).reshape((self.location_count, self.topic_count))

    def expectation(self, max_iter=10):
        lhood_old = ELBO(
            self.log_beta, self.m, self.sigma_inv, self.weight_matrix_inv, self.weight_matrix_inv_det,
            self.phi, self.zeta, self.lam, self.nu2, self.omega, self.psi2, self.locations, self.corpus
        )
        for i in range(max_iter):
            self.opt_zeta()
            self.phi = opt_phi(self.log_beta, self.lam, self.corpus)
            self.opt_zeta()
            self.opt_lam()
            self.opt_zeta()
            self.opt_nu2()
            self.opt_zeta()

            self.opt_omega()
            self.opt_psi2()

            lhood = ELBO(
                self.log_beta, self.m, self.sigma_inv, self.weight_matrix_inv, self.weight_matrix_inv_det,
                self.phi, self.zeta, self.lam, self.nu2, self.omega, self.psi2, self.locations, self.corpus
            )
            if ((lhood_old - lhood) / lhood_old) < 1e-6:
                break
            lhood_old = lhood

    def maximization(self):
        self.m = np.sum(self.omega, axis=-1) / self.topic_count

        for location_index in range(self.location_count):
            N = np.sum(self.locations == location_index)
            nu2 = self.nu2[self.locations == location_index]
            lams = self.lam[self.locations == location_index]
            omega = self.omega[location_index]

            term1 = np.diag(np.sum(nu2, axis=0))

            term2 = 0
            for lam in lams:
                lam_minus_omega = lam - omega
                term2 += np.outer(lam_minus_omega, lam_minus_omega)

            self.sigma[location_index] = (term1 + term2) / N
            self.sigma_inv[location_index] = np.linalg.inv(self.sigma[location_index])
            self.sigma_inv_det[location_index] = np.linalg.det(self.sigma_inv[location_index])

        corpus = self.corpus
        phis = self.phi

        beta_ss = np.zeros((self.topic_count, self.vocab_size))
        for doc, phi in zip(corpus, phis):
            for i in range(self.topic_count):
                for n, word in enumerate(doc):
                    beta_ss[i, word] += phi[n, i]

        log_beta = np.zeros((self.topic_count, self.vocab_size))
        for i in range(self.topic_count):
            sum_term = sum(beta_ss[i])

            if sum_term == 0:
                sum_term = (-1000) * self.vocab_size
                print(sum_term)
            else:
                sum_term = np.log(sum_term)

            for j in range(self.vocab_size):
                log_beta[i, j] = np.log(beta_ss[i, j]) - sum_term

        self.beta = np.exp(log_beta)
        self.log_beta = log_beta


def main():
    # Load the "suzhou_sense_for_gtm.json"
    documents = json.load(open('suzhou_sense_for_gtm.json', 'r', encoding='UTF-8'))

    # Create a dictionary from the documents
    dictionary = Dictionary(documents)
    dictionary.save_as_text('dictionary.txt')

    # Create a corpus from the documents
    corpus = [dictionary.doc2idx(doc) for doc in documents]

    # Save the corpus
    dictionary.save_as_text('corpus.txt')

    location_count = 5

    # Generate a random location for each document
    locations = np.random.randint(0, location_count, len(documents))

    # Generate a random coordinate for each location
    coords = np.random.uniform(0, 1, (location_count, 2))

    # Calculate the distance matrix using the coordinates and pdist
    distance_matrix = squareform(pdist(coords))

    # Calculate the weight matrix using the distance matrix and gaussian kernel
    weight_matrix = np.exp(-distance_matrix ** 2)

    # Initialize the GTM model
    gtm = GTM(10, len(dictionary), location_count, np.float64(weight_matrix))

    # Train the GTM model
    gtm.train(typed.List(corpus), locations)

    # Print the topic words according to the trained GTM model, using beta
    for i in range(10):
        print(gtm.beta[i].argsort()[-10:][::-1])

        # From idx to word
        for idx in gtm.beta[i].argsort()[-10:][::-1]:
            print(dictionary[idx], end=' ')

    # Print the topic distribution of each document
    for i in range(len(documents)):
        print(gtm.lam[i].argmax(axis=1))


if __name__ == '__main__':
    main()
