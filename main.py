import json

import numpy as np
from numpy.linalg import det
from numba import njit, prange
from numba import types, typed

from scipy.optimize import minimize

from gensim.corpora import Dictionary
from scipy.spatial.distance import pdist, squareform

np.random.seed(10)


@njit(parallel=True)
def entropy_phi(phis):
    D = len(phis)
    term = 0
    for document_index in prange(D):
        phi = phis[document_index]
        term += np.sum(phi * np.log(phi))
    return -term


@njit()
def entropy_nu2(nu2s):
    return 0.5 * np.sum(np.log(nu2s) + np.log(2 * np.pi) + 1)


@njit()
def entropy_psi2(psi2s):
    return 0.5 * np.sum(np.log(psi2s) + np.log(2 * np.pi) + 1)


@njit(parallel=True)
def log_p_w(log_beta, phis: types.ListType(types.float64[:, ::1]), corpus):
    D = len(phis)
    value = 0
    for document_index in prange(D):
        phi = phis[document_index]
        for n, word in enumerate(corpus[document_index]):
            value += np.dot(phi[n], log_beta[:, word])
    return value


@njit(parallel=True)
def log_p_z(phis, zetas, lams, nu2s):
    D = len(phis)
    value = 0
    for document_index in prange(D):
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


@njit(parallel=True)
def log_p_eta(sigma_invs, lams, nu2s, psi2s, omegas, locations):
    D = len(lams)
    T = lams.shape[1]

    value = 0
    for document_index in prange(D):
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


@njit()
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


@njit(parallel=True)
def ELBO_lam(sigma_invs, phis, zetas, lams, nu2s, omegas, locations):
    D = len(phis)

    log_p_eta = 0
    log_p_z = 0
    for document_index in prange(D):
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


@njit(parallel=True)
def df_lam(sigma_invs, phis, zetas, lams, nu2s, omegas, locations):
    D = len(phis)
    T = lams.shape[1]

    term1 = np.zeros((D, T))
    word_counts = np.zeros(D)
    for document_index in prange(D):
        term1[document_index] = np.sum(phis[document_index], axis=0)
        word_counts[document_index] = len(phis[document_index])

    # Temporal solution for the bug of numba
    word_counts_divided_by_zetas = np.empty((D, 1))
    word_counts_divided_by_zetas[:, 0] = word_counts / zetas

    term2 = - word_counts_divided_by_zetas * np.exp(lams + nu2s / 2)

    term3 = np.zeros((D, T))
    for document_index in prange(D):
        sigma_inv = sigma_invs[locations[document_index]]
        lam = lams[document_index]
        omega = omegas[locations[document_index]]
        term3[document_index] = - np.dot(sigma_inv, lam - omega)

    return term1 + term2 + term3


# Parallel makes it slower, as task is too small, overhead spent on parallelization is too large.
@njit()
def numba_logsumexp_stable(p):
    n, m = p.shape
    out = np.empty((n,))
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

    return out


@njit(parallel=True)
# @njit()
def opt_phi(log_beta, lams, corpus):
    D = len(lams)
    T = lams.shape[1]

    phis = typed.List.empty_list(types.float64[:, ::1])
    for _ in range(D):
        phis.append(np.empty((0, 0)))

    for document_index in prange(D):
        lam = lams[document_index]
        N = len(corpus[document_index])
        log_phi = np.zeros((N, T), dtype=np.float64)

        for n, word in enumerate(corpus[document_index]):
            log_phi[n, :] = lam + log_beta[:, word]

        # Using Numba operations for log-sum-exp
        log_phi_sum = np.empty((N, 1))
        log_phi_sum[:, 0] = numba_logsumexp_stable(log_phi)

        # This line cause the process terminated unexpectedly. This should be the bug of numba.
        # log_phi_sum_ = log_phi_sum[:, np.newaxis]

        # Vectorized calculation of phi
        phi = np.exp(log_phi - log_phi_sum)

        phis[document_index] = phi
        # phis.append(phi)

    return phis


@njit(parallel=True)
def opt_nu2(sigma_invs, zetas, lams, locations, word_counts):
    D = len(lams)
    T = lams.shape[1]

    # Precompute constants and reshape for batch operations
    sigma_inv_diag = np.zeros((D, T))
    for document_index in prange(D):
        sigma_inv_diag[document_index] = np.diag(sigma_invs[locations[document_index]])
    term1_nu2 = -0.5 * sigma_inv_diag

    # Initialize variables for batch optimization
    init_x = np.ones((D, T)) * 10
    x = init_x
    log_x = np.log(x)

    # Temporal solution for the bug of numba
    word_counts_divided_by_zetas = np.empty((D, 1))
    word_counts_divided_by_zetas[:, 0] = word_counts / zetas

    while True:
        x = np.exp(log_x)

        # Batch calculations for df_nu2 and df2_nu2
        term2_nu2 = -0.5 * word_counts_divided_by_zetas * np.exp(lams + x / 2)
        term3_nu2 = 1 / (2 * x)
        df1 = term1_nu2 + term2_nu2 + term3_nu2

        term1_nu2_2 = -0.25 * word_counts_divided_by_zetas * np.exp(lams + x / 2)
        term2_nu2_2 = -0.5 * (1 / (x * x))
        df2 = term1_nu2_2 + term2_nu2_2

        log_x -= (x * df1) / (x * x * df2 + x * df1)

        # Check for convergence and NaN handling
        if np.all(np.abs(df1) <= 0.0001):
            break
        if np.any(np.isnan(x)):
            init_x *= 10
            log_x = np.log(init_x)

    return x


@njit(parallel=True)
def maximize_sigmas(lams, nu2s, omegas, locations):
    location_count = len(omegas)
    topic_count = lams.shape[1]

    sigmas = np.zeros((location_count, topic_count, topic_count))

    for location_index in prange(location_count):
        N = np.sum(locations == location_index)
        nu2s_location = nu2s[locations == location_index]
        lams_location = lams[locations == location_index]
        omega = omegas[location_index]

        term1 = np.diag(np.sum(nu2s_location, axis=0))

        term2 = np.zeros((topic_count, topic_count))
        for lam in lams_location:
            lam_minus_omega = lam - omega
            term2 += np.outer(lam_minus_omega, lam_minus_omega)

        sigmas[location_index] = (term1 + term2) / N

    return sigmas


@njit(parallel=True)
def maximize_beta(phis, corpus, vocab_size):
    topic_count = phis[0].shape[1]

    beta_ss = np.zeros((topic_count, vocab_size))

    for document_index in prange(len(corpus)):
        phi = phis[document_index]
        for n, word in enumerate(corpus[document_index]):
            beta_ss[:, word] += phi[n, :]

    # Temporal solution for the bug of numba
    sum_log_term = np.zeros((topic_count, 1))
    sum_log_term[:, 0] = np.log(np.sum(beta_ss, axis=1))

    beta_log = np.log(beta_ss)
    beta_log_normed = np.exp(beta_log - sum_log_term)

    return np.exp(beta_log_normed), beta_log_normed


class GTM:
    def __init__(self, topic_count, vocab_size, location_count, weight_matrix):
        # Model Property
        self.topic_count = topic_count
        self.vocab_size = vocab_size
        self.location_count = location_count

        # Model parameter
        self.weight_matrix = weight_matrix
        self.weight_matrix_inv = None
        self.weight_matrix_inv_det = None
        self.m = None
        self.sigma = None
        self.sigma_inv = None
        self.sigma_inv_det = None
        self.beta = None
        self.log_beta = None

        # Training parameters
        self.corpus = typed.List.empty_list(types.int32[::1])
        self.locations = None
        self.document_size = 0
        self.word_counts = None

        # Variational factors
        self.phi = typed.List.empty_list(types.float64[:, ::1])
        self.zeta = None
        self.lam = None
        self.nu2 = None
        self.omega = None
        self.psi2 = None

        self.init_model_param()

    def init_model_param(self):
        # Size(location_count)
        self.m = np.zeros(self.location_count)
        # Size(location_count, topic_count, topic_count)
        self.sigma = np.tile(np.eye(self.topic_count), (self.location_count, 1, 1))
        self.sigma_inv = self.sigma
        self.sigma_inv_det = np.ones(self.location_count)
        # Size(topic_count, vocab_size)
        self.beta = 0.001 + np.random.uniform(0, 1, (self.topic_count, self.vocab_size))
        self.beta /= np.sum(self.beta, axis=1)[:, np.newaxis]
        self.log_beta = np.log(self.beta)

        self.weight_matrix_inv = np.linalg.inv(self.weight_matrix)
        self.weight_matrix_inv_det = np.linalg.det(self.weight_matrix_inv)

    def init_variational_factor(self):
        self.zeta = np.ones(self.document_size) * 10

        self.lam = np.zeros((self.document_size, self.topic_count))
        self.nu2 = np.ones((self.document_size, self.topic_count))

        self.omega = np.zeros((self.location_count, self.topic_count))
        self.psi2 = np.ones((self.location_count, self.topic_count))

        # The size of phi is dependent on the words count in each document
        phi_init_value = 1 / self.topic_count
        for i in range(self.document_size):
            self.phi.append(
                np.ones((self.word_counts[i], self.topic_count)) * phi_init_value
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

    def expectation(self, max_iter=10):
        likelihood_old = ELBO(
            self.log_beta, self.m, self.sigma_inv, self.weight_matrix_inv, self.weight_matrix_inv_det,
            self.phi, self.zeta, self.lam, self.nu2, self.omega, self.psi2, self.locations, self.corpus
        )
        for i in range(max_iter):
            self.opt_zeta()
            self.phi = opt_phi(self.log_beta, self.lam, self.corpus)
            self.opt_zeta()
            self.opt_lam()
            self.opt_zeta()
            self.nu2 = opt_nu2(self.sigma_inv, self.zeta, self.lam, self.locations, self.word_counts)
            self.opt_zeta()

            self.opt_omega()
            self.opt_psi2()

            likelihood = ELBO(
                self.log_beta, self.m, self.sigma_inv, self.weight_matrix_inv, self.weight_matrix_inv_det,
                self.phi, self.zeta, self.lam, self.nu2, self.omega, self.psi2, self.locations, self.corpus
            )
            if ((likelihood_old - likelihood) / likelihood_old) < 1e-6:
                break
            likelihood_old = likelihood

    def opt_zeta(self):
        self.zeta = np.sum(np.exp(self.lam + self.nu2 / 2), axis=-1)

    def opt_lam(self):
        fn = lambda x: - ELBO_lam(self.sigma_inv, self.phi, self.zeta,
                                  x.reshape((self.document_size, self.topic_count)), self.nu2, self.omega,
                                  self.locations)
        g = lambda x: - df_lam(self.sigma_inv, self.phi, self.zeta, x.reshape((self.document_size, self.topic_count)),
                               self.nu2, self.omega, self.locations).flatten()

        res = minimize(fn, x0=self.lam.flatten(), jac=g, method='Newton-CG', options={'disp': 0})
        lam_optimized = res.x

        self.lam = lam_optimized.reshape((self.document_size, self.topic_count))

    def df_omega(self, omega):
        # Vectorized computation for location loop
        omega_d_location = np.zeros((self.location_count, self.topic_count))
        for location_index in range(self.location_count):
            sigma_inv = self.sigma_inv[location_index]
            document_indices = np.nonzero(self.locations == location_index)[0]
            lam = self.lam[document_indices]
            lam_minus_omega = lam - omega[location_index]
            omega_d_location[location_index] = np.dot(sigma_inv, lam_minus_omega.sum(axis=0))

        # Vectorized computation for topic loop
        omega_minus_m = omega - self.m[:, np.newaxis]
        omega_d_topic = -np.dot(self.weight_matrix_inv, omega_minus_m)

        return omega_d_location + omega_d_topic

    def ELBO_omega(self, omegas):
        log_p_doc = 0
        for location_index in range(self.location_count):
            lam = self.lam[self.locations == location_index]
            omega = omegas[location_index]
            lam_minus_omega = lam - omega
            sigma_inv = self.sigma_inv[location_index]

            term3 = 0.5 * np.dot(lam_minus_omega, sigma_inv) * lam_minus_omega
            log_p_doc += -np.sum(term3)

        log_p_topic = 0

        weight_matrix_inv = self.weight_matrix_inv
        m = self.m
        omega_minus_m = omegas - m[:, np.newaxis]

        term3 = 0.5 * np.dot(omega_minus_m.T, weight_matrix_inv).T * omega_minus_m
        log_p_topic += -np.sum(term3)

        return log_p_doc + log_p_topic

    def opt_omega(self):
        omega = self.omega.flatten()

        fn = lambda x: - self.ELBO_omega(x.reshape((self.location_count, self.topic_count)))
        g = lambda x: - self.df_omega(x.reshape((self.location_count, self.topic_count))).flatten()

        res = minimize(fn, x0=omega, jac=g, method='Newton-CG', options={'disp': 0})
        omega_optimized = res.x

        self.omega = omega_optimized.reshape((self.location_count, self.topic_count))

    def df_psi2(self, psi2):
        term1 = 0.5 * np.diagonal(self.sigma_inv, axis1=1, axis2=2)
        term2 = 0.5 * np.diag(self.weight_matrix_inv)

        psi2_d = - (term1 + term2[:, np.newaxis])

        term3 = 1 / (2 * psi2)
        psi2_d += term3

        return psi2_d

    def df2_psi2(self, psi2):
        return -0.5 * (1 / (psi2 * psi2))

    def opt_psi2(self):
        g = lambda psi2: self.df_psi2(psi2)
        h = lambda psi2: self.df2_psi2(psi2)

        init_x = np.ones((self.location_count, self.topic_count)) * 10
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

        self.psi2 = np.exp(log_x)

    def maximization(self):
        self.m = np.sum(self.omega, axis=-1) / self.topic_count

        self.sigma = maximize_sigmas(self.lam, self.nu2, self.omega, self.locations)
        self.sigma_inv = np.linalg.inv(self.sigma)
        self.sigma_inv_det = np.linalg.det(self.sigma_inv)

        self.beta, self.log_beta = maximize_beta(self.phi, self.corpus, self.vocab_size)

    def save(self, path):
        np.savez(
            path,

        )

    def load(self, path):
        data = np.load(path)
        self.beta = data['arr_0']
        self.m = data['arr_1']
        self.sigma = data['arr_2']
        self.omega = data['arr_3']
        self.psi2 = data['arr_4']


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


if __name__ == '__main__':
    main()
