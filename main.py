import json

import numpy as np
from numpy.linalg import det
from numba.experimental import jitclass
from numba import types, typed

from scipy.optimize import minimize

from gensim.corpora import Dictionary
from scipy.spatial.distance import pdist, squareform

np.random.seed(10)


@jitclass()
class GTM:
    topic_count: types.int32
    vocab_size: types.int32
    location_count: types.int32
    weight_matrix: types.float32[:, :]
    weight_matrix_inv: types.float32[:, :]
    weight_matrix_inv_det: types.float32
    m: types.float32[:]
    sigma: types.float32[:, :, :]
    sigma_inv: types.float32[:, :, :]
    sigma_inv_det: types.float32[:]
    beta: types.float32[:, :]
    log_beta: types.float32[:, :]
    corpus: types.ListType(types.int32[::1])
    locations: types.int32[:]
    document_size: types.int32
    word_counts: types.int32[:]
    phi: types.ListType(types.float32[:, :])
    zeta: types.float32[:]
    lam: types.float32[:, :]
    nu2: types.float32[:, :]
    omega: types.float32[:, :]
    psi2: types.float32[:, :]

    def __init__(self, topic_count, vocab_size, location_count, weight_matrix):
        # Model Property
        self.topic_count = topic_count
        self.vocab_size = vocab_size
        self.location_count = location_count

        # Model parameter
        self.weight_matrix = weight_matrix
        self.weight_matrix_inv = np.empty((0, 0), dtype=np.float32)
        self.weight_matrix_inv_det = np.float32(0)
        self.m = np.empty((0,), dtype=np.float32)
        self.sigma = np.empty((0, 0, 0), dtype=np.float32)
        self.sigma_inv = np.empty((0, 0, 0), dtype=np.float32)
        self.sigma_inv_det = np.empty((0,), dtype=np.float32)
        self.beta = np.empty((0, 0), dtype=np.float32)
        self.log_beta = np.empty((0, 0), dtype=np.float32)

        # Training parameters
        self.corpus = typed.List.empty_list(types.int32[::1])
        self.locations = np.empty((0, ), dtype=np.int32)
        self.document_size = 0
        self.word_counts = np.empty((0, ), dtype=np.int32)

        # Variational factors
        self.phi = typed.List.empty_list(types.float32[:, :])
        self.zeta = np.empty((0, ), dtype=np.float32)
        self.lam = np.empty((0, 0), dtype=np.float32)
        self.nu2 = np.empty((0, 0), dtype=np.float32)
        self.omega = np.empty((0, 0), dtype=np.float32)
        self.psi2 = np.empty((0, 0), dtype=np.float32)

        self.init_model_param()

    def init_model_param(self):
        # Size(location_count)
        self.m = np.zeros(self.location_count, dtype=np.float32)
        # Size(location_count, topic_count, topic_count)
        self.sigma = np.eye(self.topic_count, dtype=np.float32).repeat(self.location_count).reshape((self.location_count, self.topic_count, self.topic_count))
        self.sigma_inv = self.sigma
        self.sigma_inv_det = np.ones(self.location_count, dtype=np.float32)
        # Size(topic_count, vocab_size)
        self.beta = np.float32(0.001) + np.random.uniform(0, 1, (self.topic_count, self.vocab_size)).astype(np.float32)
        self.log_beta = np.log(self.beta)

        # for i in range(self.topic_count):
        #     beta[i] /= sum(beta[i])
        self.beta /= np.sum(self.beta, axis=1)[:, np.newaxis]

        self.weight_matrix_inv = np.linalg.inv(self.weight_matrix)
        self.weight_matrix_inv_det = np.linalg.det(self.weight_matrix_inv)

    def init_variational_factor(self):
        self.zeta = np.ones(self.document_size, dtype=np.float32) * 10

        self.lam = np.zeros((self.document_size, self.topic_count), dtype=np.float32)
        self.nu2 = np.ones((self.document_size, self.topic_count), dtype=np.float32)

        self.omega = np.zeros((self.location_count, self.topic_count), dtype=np.float32)
        self.psi2 = np.ones((self.location_count, self.topic_count), dtype=np.float32)

        # The size of phi is dependent on the words count in each document
        phi_init_value = np.float32(1 / self.topic_count)
        for i in range(self.document_size):
            self.phi.append(
                np.ones((self.word_counts[i], self.topic_count), dtype=np.float32) * phi_init_value
            )

    def train(self, corpus, locations, max_iter=1000):
        self.corpus = corpus
        self.locations = locations
        self.document_size = len(corpus)
        self.word_counts = [len(doc) for doc in corpus]
        self.init_variational_factor()

        after = self.lhood_bnd_total()
        for _ in range(max_iter):
            before = after
            self.expectation()
            self.maximization()
            after = self.lhood_bnd_total()

            print('lhood = ', after)
            print(((before - after) / before))
            if ((before - after) / before) < 0.001:
                break

    def entropy_per_document(self, document_index):
        # Calculate the entropy of the document
        nu2 = self.nu2[document_index]
        term1 = .5 * sum(np.log(nu2) + np.log(2 * np.pi) + 1)

        phi = self.phi[document_index]
        term2 = sum(np.dot(phi[n], np.log(phi[n])) for n in range(self.word_counts[document_index]))

        return term1 - term2

    def entropy_per_location(self, location_index):
        # Calculate the entropy of the document belonging to the location
        document_indices = np.nonzero(self.locations == location_index)[0]
        document_entropy = 0
        for document_index in document_indices:
            nu2 = self.nu2[document_index]
            term1 = .5 * sum(np.log(nu2) + np.log(2 * np.pi) + 1)

            phi = self.phi[document_index]
            term2 = 0
            for n in range(self.word_counts[document_index]):
                term2 += np.dot(phi[n], np.log(phi[n]))

            document_entropy += term1 - term2

        # document_entropy = sum(self.entropy_per_document(document_index) for document_index in document_indices)

        # Calculate the entropy of the location
        psi2 = self.psi2[location_index]
        term = .5 * sum(np.log(psi2) + np.log(2 * np.pi) + 1)

        return document_entropy + term

    def lhood_bnd_per_document(self, document_index, lam=None):
        omega = self.omega[self.locations[document_index]]
        psi2_diag = np.diag(self.psi2[self.locations[document_index]])
        sigma_inv = self.sigma_inv[self.locations[document_index]]

        zeta = self.zeta[document_index]
        phi = self.phi[document_index]
        if lam is None:
            lam = self.lam[document_index]
        nu2 = self.nu2[document_index]
        nu2_diag = np.diag(nu2)

        topic_count = self.topic_count
        lam_minus_omega = lam - omega

        term1 = 0.5 * np.log(det(self.sigma_inv[self.locations[document_index]]))
        term2 = 0.5 * topic_count * np.log(2 * np.pi)
        term3 = 0.5 * (np.trace(np.dot(nu2_diag, sigma_inv)) + np.trace(np.dot(psi2_diag, sigma_inv)) + np.dot(np.dot(lam_minus_omega.T, sigma_inv), lam_minus_omega))
        log_p_eta = term1 - term2 - term3

        term2 = (1 / zeta) * sum(np.exp(lam + nu2 / 2))
        term3 = 1 - np.log(zeta)
        log_p_zn = self.word_counts[document_index] * (-term2 + term3)

        log_p_wn = 0
        for n, word in enumerate(self.corpus[document_index]):
            term1 = np.dot(lam, phi[n])
            log_p_zn += term1

            log_p_wn += np.dot(phi[n], self.log_beta[:, word])

        return log_p_eta + log_p_wn + log_p_zn + self.entropy_per_document(document_index)

    def lhood_bnd_total(self, omega_v=None):
        if omega_v is not None:
            # Help numba to infer the type of omega_v
            if not isinstance(omega_v, np.ndarray):
                omega_v = np.array(omega_v)
            omega_v = omega_v.reshape((self.location_count, self.topic_count))
            omegas = omega_v
        else:
            omegas = self.omega

        log_p_sum_doc = 0

        for location_index in range(self.location_count):
            omega = omegas[location_index]
            psi2_diag = np.diag(self.psi2[location_index])

            sigma_inv = self.sigma_inv[location_index]
            log_beta = np.log(self.beta)

            document_indices = np.nonzero(self.locations == location_index)[0]
            for document_index in document_indices:

                zeta = self.zeta[document_index]
                phi = self.phi[document_index]
                lam = self.lam[document_index]
                nu2 = self.nu2[document_index]
                nu2_diag = np.diag(nu2)

                topic_count = self.topic_count
                lam_minus_omega = lam - omega

                term1 = 0.5 * np.log(det(sigma_inv))
                term2 = 0.5 * topic_count * np.log(2 * np.pi)
                term3 = 0.5 * (np.trace(np.dot(nu2_diag, sigma_inv)) + np.trace(np.dot(psi2_diag, sigma_inv)) + np.dot(np.dot(lam_minus_omega.T, sigma_inv), lam_minus_omega))
                log_p_eta = term1 - term2 - term3

                term2 = (1 / zeta) * sum(np.exp(lam + nu2 / 2))
                term3 = 1 - np.log(zeta)
                log_p_zn = self.word_counts[document_index] * (-term2 + term3)

                log_p_wn = 0
                for n, word in enumerate(self.corpus[document_index]):
                    term1 = np.dot(lam, phi[n])
                    log_p_zn += term1

                    log_p_wn += np.dot(phi[n], log_beta[:, word])

                log_p_sum_doc += log_p_eta + log_p_wn + log_p_zn

        # Calculate the log_p_mu
        log_p_mu = 0
        for i in range(self.topic_count):
            psi2 = self.psi2[:, i]
            omega_minus_m = omegas[:, i] - self.m

            term1 = 0.5 * np.log(self.weight_matrix_inv_det)
            term2 = 0.5 * self.location_count * np.log(2 * np.pi)
            term3 = 0.5 * (np.trace(np.dot(np.diag(psi2), self.weight_matrix_inv)) + np.dot(
                np.dot(omega_minus_m.T, self.weight_matrix_inv), omega_minus_m))
            log_p_mu += term1 - term2 - term3

        return log_p_sum_doc + log_p_mu + sum([self.entropy_per_location(location_index) for location_index in range(self.location_count)])

    def lhood_bnd_involved_omega(self, omega_v):
        omega_v = omega_v.reshape((self.location_count, self.topic_count))
        omegas = omega_v

        log_p_doc = 0

        for document_index in range(self.document_size):
            location_index = self.locations[document_index]
            omega = omegas[location_index]
            lam = self.lam[document_index]

            sigma_inv = self.sigma_inv[location_index]
            lam_minus_omega = lam - omega

            term3 = 0.5 * np.dot(np.dot(lam_minus_omega.T, sigma_inv), lam_minus_omega)

            log_p_doc += -term3

        log_p_topic = 0

        for topic_index in range(self.topic_count):
            omega = omegas[:, topic_index]
            weight_matrix_inv = self.weight_matrix_inv
            m = self.m

            omega_minus_m = omega - m

            term3 = 0.5 * np.dot(np.dot(omega_minus_m.T, weight_matrix_inv), omega_minus_m)

            log_p_topic += -term3

        return log_p_doc + log_p_topic

    def df_lam(self, document_index, lam):
        nu2 = self.nu2[document_index]
        zeta = self.zeta[document_index]
        omega = self.omega[self.locations[document_index]]
        phi = self.phi[document_index]

        sigma_inv = self.sigma_inv[self.locations[document_index]]
        lam_minus_omega = lam - omega

        term1 = np.dot(sigma_inv, lam_minus_omega)

        term2 = np.sum(phi, axis=0)

        N = self.word_counts[document_index]
        term3 = (N / zeta) * (np.exp(lam + nu2 / 2))

        return -term1 + term2 - term3

    def opt_lam(self, document_index):
        lam = self.lam[document_index]

        # TODO: Optimize in batch for all document in a location.
        fn = lambda x: -self.lhood_bnd_per_document(document_index, x)
        g = lambda x: -self.df_lam(document_index, x)

        res = minimize(fn, x0=lam, jac=g, method='Newton-CG', options={'disp': 0})
        lam_optimized = res.x

        self.lam[document_index] = lam_optimized

    def df_nu2(self, document_index, nu2):
        N = self.word_counts[document_index]
        sigma_inv = self.sigma_inv[self.locations[document_index]]
        zeta = self.zeta[document_index]
        lam = self.lam[document_index]

        term1 = 0.5 * np.diag(sigma_inv)
        term2 = 0.5 * (N / zeta) * np.exp(lam + nu2 / 2)
        term3 = 1 / (2 * nu2)
        return -term1 - term2 + term3

    def df2_nu2(self, document_index, nu2):
        N = self.word_counts[document_index]
        zeta = self.zeta[document_index]
        lam = self.lam[document_index]

        term1 = 0.25 * (N / zeta) * np.exp(lam + nu2 / 2)
        term2 = 0.5 * (1 / (nu2 * nu2))
        return -term1 - term2

    def opt_nu2(self, document_index):
        g = lambda nu2: self.df_nu2(document_index, nu2)
        h = lambda nu2: self.df2_nu2(document_index, nu2)

        init_x = np.ones(self.topic_count) * 10
        x = init_x

        log_x = np.log(x)
        df1 = np.ones(self.topic_count)

        while np.all(np.abs(df1) > 0.0001):
            if np.any(np.isnan(x)):
                init_x = init_x * 10
                x = init_x
                log_x = np.log(x)
            x = np.exp(log_x)

            df1 = g(x)
            df2 = h(x)

            log_x -= (x * df1) / (x * x * df2 + x * df1)

        self.nu2[document_index] = np.exp(log_x)

    def opt_zeta(self, document_index):
        lam = self.lam[document_index]
        nu2 = self.nu2[document_index]
        self.zeta[document_index] = sum(np.exp(lam + nu2 / 2)) + 1

    def log_sum(self, log_a, log_b):
        """
        Explain its functionality
        """
        if log_a < log_b:
            return log_b + np.log(1 + np.exp(log_a - log_b))
        else:
            return log_a + np.log(1 + np.exp(log_b - log_a))

    def opt_phi(self, document_index):
        phi = self.phi[document_index]
        lam = self.lam[document_index]

        log_beta = self.log_beta

        log_phi = np.zeros_like(phi)

        for n, word in enumerate(self.corpus[document_index]):
            log_phi_sum = 0
            for i in range(self.topic_count):
                log_phi[n, i] = lam[i] + log_beta[i, word]
                if i == 0:
                    log_phi_sum = log_phi[n, i]
                else:
                    log_phi_sum = self.log_sum(log_phi_sum, log_phi[n, i])

            for i in range(self.topic_count):
                log_phi[n, i] -= log_phi_sum
                phi[n, i] = np.exp(log_phi[n, i])

        self.phi[document_index] = phi

    def df_omega(self, omega_v):
        if omega_v.size == 1:
            omega_single = omega_v
            omega_v = self.omega
            omega_v[0, 0] = omega_single

        omega_v = omega_v.reshape((self.location_count, self.topic_count))
        omega_d = np.zeros((self.location_count, self.topic_count))

        for location_index in range(self.location_count):
            omega = omega_v[location_index]
            sigma_inv = self.sigma_inv[location_index]

            term1 = 0
            for document_index in np.nonzero(self.locations == location_index)[0]:
                lam = self.lam[document_index]
                lam_minus_omega = lam - omega
                term1 += np.dot(sigma_inv, lam_minus_omega)

            omega_d[location_index] += term1

        for topic_index in range(self.topic_count):
            omega = omega_v[:, topic_index]
            weight_matrix_inv = self.weight_matrix_inv
            m = self.m

            term2 = np.dot(weight_matrix_inv, omega - m)

            omega_d[:, topic_index] += -term2

        return omega_d

    def opt_omega(self):
        omega = self.omega.flatten()

        fn = lambda x: - self.lhood_bnd_involved_omega(x)
        g = lambda x: - self.df_omega(x).flatten()

        res = minimize(fn, x0=omega, jac=g, method='Newton-CG', options={'disp': 0})
        omega_optimized = res.x

        self.omega = omega_optimized.reshape((self.location_count, self.topic_count))

    def df_psi2(self, psi2):
        psi2_d = np.zeros((self.location_count, self.topic_count))

        for location_index in range(self.location_count):
            sigma_inv = self.sigma_inv[location_index]
            term1 = 0.5 * np.diag(sigma_inv)
            psi2_d[location_index] += term1

        for topic_index in range(self.topic_count):
            weight_matrix_inv = self.weight_matrix_inv
            term2 = 0.5 * np.diag(weight_matrix_inv)
            psi2_d[:, topic_index] += term2

        psi2_d = -psi2_d

        psi2 = psi2.reshape((self.location_count, self.topic_count))
        term3 = 1 / (2 * psi2)
        psi2_d += term3

        return psi2_d

    def df2_psi2(self, psi2):
        psi2 = psi2.reshape((self.location_count, self.topic_count))
        return - 0.5 * (1 / (psi2 * psi2))

    def opt_psi2(self):
        g = lambda psi2: self.df_psi2(psi2).flatten()
        h = lambda psi2: self.df2_psi2(psi2).flatten()

        init_x = np.ones((self.location_count, self.topic_count)).flatten() * 10
        x = init_x

        log_x = np.log(x)
        df1 = np.ones((self.location_count, self.topic_count)).flatten()

        while np.all(np.abs(df1) > 0.0001):
            if np.any(np.isnan(x)):
                init_x = init_x * 10
                x = init_x
                log_x = np.log(x)
            x = np.exp(log_x)

            df1 = g(x)
            df2 = h(x)

            log_x -= (x * df1) / (x * x * df2 + x * df1)

        self.psi2 = np.exp(log_x).reshape((self.location_count, self.topic_count))

    def expectation(self, max_iter=10):
        lhood_old = self.lhood_bnd_total()
        for i in range(max_iter):
            for document_index in range(self.document_size):
                self.opt_zeta(document_index)
                self.opt_phi(document_index)
                self.opt_zeta(document_index)
                self.opt_lam(document_index)
                self.opt_zeta(document_index)
                self.opt_nu2(document_index)
                self.opt_zeta(document_index)

            self.opt_omega()
            self.opt_psi2()

            lhood = self.lhood_bnd_total()
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
    gtm = GTM(10, len(dictionary), location_count, np.float32(weight_matrix))

    # Train the GTM model
    gtm.train(corpus, locations)

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
