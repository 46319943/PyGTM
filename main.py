import numpy as np
from numpy.linalg import det
from numpy.linalg import inv
from scipy.optimize import minimize
import os
from os import path
from gensim.corpora import Dictionary

np.random.seed(10)


class GTM():

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
        self.corpus = None
        self.locations = None
        self.document_size = None
        self.word_counts = None

        # Variational factors
        self.phi = None
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
        self.sigma_inv_det = np.ones((self.location_count))
        # Size(topic_count, vocab_size)
        self.beta = 0.001 + np.random.uniform(0, 1, (self.topic_count, self.vocab_size))
        self.log_beta = np.log(self.beta)

        # for i in range(self.topic_count):
        #     beta[i] /= sum(beta[i])
        self.beta /= np.sum(self.beta, axis=1)[:, np.newaxis]

        self.weight_matrix_inv = np.linalg.inv(self.weight_matrix)
        self.weight_matrix_inv_det = np.linalg.det(self.weight_matrix_inv)

    def init_variational_factor(self):
        self.zeta = np.empty(self.document_size)
        self.zeta = 10

        self.lam = np.zeros((self.document_size, self.topic_count))
        self.nu2 = np.ones((self.document_size, self.topic_count))

        self.omega = np.zeros((self.location_count, self.topic_count))
        self.psi2 = np.ones((self.location_count, self.topic_count))

        # The size of phi is dependent on the words count in each document
        self.phi = [None] * self.document_size
        phi_init_value = 1 / self.topic_count
        for i in range(self.document_size):
            self.phi[i] = np.ones((self.word_counts[i], self.topic_count)) * phi_init_value

    def train(self, corpus, locations, max_iter=1000):
        self.corpus = corpus
        self.locations = locations
        self.document_size = len(corpus)
        self.word_counts = [len(doc) for doc in corpus]
        self.init_variational_factor()

        for _ in range(max_iter):
            self.expectation(corpus, locations)
            self.maximization(corpus, locations)

    def entropy_per_location(self, location_index):
        # Calculate the entropy of the document belonging to the location
        document_indices = np.nonzero(self.locations == location_index)
        document_entropy = 0
        for document_index in document_indices:
            nu2 = self.nu2[document_index]
            term1 = .5 * sum(np.log(nu2) + np.log(2 * np.pi) + 1)

            phi = self.phi[document_index]
            term2 = sum(np.dot(phi[n], np.log(phi[n])) for n in range(self.word_counts[document_index]))

            document_entropy += term1 - term2

        document_entropy = sum(self.entropy_per_document(document_index) for document_index in document_indices)

        # Calculate the entropy of the location
        psi2 = self.psi2[location_index]
        term = .5 * sum(np.log(psi2) + np.log(2 * np.pi) + 1)

        return document_entropy + term

    def lhood_bnd_per_location(self, location_index):
        omega = self.omega[location_index]
        psi2_diag = np.diag(self.psi2[location_index])

        sigma_inv = self.sigma_inv[location_index]
        log_beta = np.log(self.beta)

        log_p_sum_doc = 0
        document_indices = np.nonzero(self.locations == location_index)
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
            diff = self.omega[:, i] - self.m

            term1 = 0.5 * np.log(self.weight_matrix_inv_det)
            term2 = 0.5 * self.location_count * np.log(2 * np.pi)
            term3 = 0.5 * (np.trace(np.dot(np.diag(psi2), self.weight_matrix_inv)) + np.dot(
                np.dot(diff.T, self.weight_matrix_inv), diff))
            log_p_mu += term1 - term2 - term3

        return log_p_sum_doc + log_p_mu + self.entropy_per_location(location_index)

    def df_lam(self, document_index):
        lam = self.lam[document_index]
        nu2 = self.nu2[document_index]
        zeta = self.zeta[document_index]
        phi = self.phi[document_index]

        sigma_inv = self.sigma_inv[self.locations[document_index]]
        lam_minus_phi = lam - phi

        term1 = np.dot(sigma_inv, lam_minus_phi)

        term2 = np.sum(phi, axis=0)

        N = self.word_counts[document_index]
        term3 = (N / zeta) * (np.exp(lam + nu2 / 2))

        return -term1 + term2 - term3

    def opt_lam(self, document_index):
        lam = self.lam[document_index]

        # TODO: Optimize in batch for all document in a location.
        fn = lambda x: -self.lhood_bnd_per_location(self.locations[document_index])
        g = lambda x: -self.df_lam(document_index)

        res = minimize(fn, x0=lam, jac=g, method='Newton-CG', options={'disp': 0})
        lam_optimized = res.x

        self.lam[document_index] = lam_optimized


    def df_nu2(self, document_index):
        N = self.word_counts[document_index]
        sigma_inv = self.sigma_inv[document_index]
        zeta = self.zeta[document_index]
        lam = self.lam[document_index]
        nu2 = self.nu2[document_index]

        term1 = 0.5 * np.diag(sigma_inv)
        term2 = 0.5 * (N / zeta) * np.exp(lam + nu2 / 2)
        term3 = 1 / (2 * nu2)
        return -term1 - term2 + term3

    def df2_nu2(self, document_index):
        N = self.word_counts[document_index]
        zeta = self.zeta[document_index]
        lam = self.lam[document_index]
        nu2 = self.nu2[document_index]

        term1 = 0.25 * (N / zeta) * np.exp(lam + nu2 / 2)
        term2 = 0.5 * (1 / (nu2 * nu2))
        return -term1 - term2

    def opt_nu2(self, document_index):
        g = lambda _: self.df_nu2(document_index)
        h = lambda _: self.df2_nu2(document_index)

        init_x = np.ones(self.topic_count) * 10
        x = init_x

        log_x = np.log(x)
        df1 = np.ones(self.topic_count)

        while np.all(np.abs(df1) > 0.0001):
            if np.isnan(x):
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
        Explain it's fu
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

    def df_omega(self, location_index):
        pass

    def opt_omega(self, location_index):
        pass


    def inference(self, doc, mod):
        mu, sigma_inv, log_beta = mod

        topic_count = len(log_beta)
        document_size = len(doc)
        var = self.init_variational_factor(topic_count, document_size)
        zeta, phi, lam, nu2 = var

        for i in range(20):
            var = zeta, phi, lam, nu2
            lhood_old = self.lhood_bnd_per_location(doc, var, mod)
            # print ('lhood_old = ', lhood_old)

            var = zeta, phi, lam, nu2
            zeta = self.opt_zeta(doc, var, mod)

            var = zeta, phi, lam, nu2
            # print (lhood_bnd(doc, var, mod))

            var = zeta, phi, lam, nu2
            lam = self.opt_lam(doc, var, mod)
            lam[-1] = 0

            var = zeta, phi, lam, nu2
            # print (lhood_bnd(doc, var, mod))

            var = zeta, phi, lam, nu2
            zeta = self.opt_zeta(doc, var, mod)

            var = zeta, phi, lam, nu2
            # print (lhood_bnd(doc, var, mod))

            var = zeta, phi, lam, nu2
            nu2 = self.opt_nu2(doc, var, mod)

            var = zeta, phi, lam, nu2
            # print (lhood_bnd(doc, var, mod))

            var = zeta, phi, lam, nu2
            zeta = self.opt_zeta(doc, var, mod)

            var = zeta, phi, lam, nu2
            # print (lhood_bnd(doc, var, mod))

            var = zeta, phi, lam, nu2
            phi = self.opt_phi(doc, var, mod)

            var = zeta, phi, lam, nu2
            zeta = self.opt_zeta(doc, var, mod)

            var = zeta, phi, lam, nu2
            # print (lhood_bnd(doc, var, mod))

            var = zeta, phi, lam, nu2
            lhood = self.lhood_bnd_per_location(doc, var, mod)
            if ((lhood_old - lhood) / lhood_old) < 1e-6:
                break
            # print ('-lhood = ', lhood)

            lhood_old = lhood

        return zeta, phi, lam, nu2

    def expectation(self, corpus, mod):
        corpus_var = []
        for d, doc in enumerate(corpus):
            var = self.inference(doc, mod)
            zeta, phi, lam, nu2 = var
            corpus_var.append((zeta, phi.copy(), lam.copy(), nu2.copy()))
        return corpus_var

    def maximization(self, corpus, corpus_var, vocab_size):
        lams = []
        nu2s = []
        phis = []
        for zeta, phi, lam, nu2 in corpus_var:
            lams.append(lam)
            nu2s.append(nu2)
            phis.append(phi)

        mu_sum = sum(lams)
        mu = mu_sum / len(corpus)

        sigma_sum = sum(np.diag(nu2) + np.outer(lam - mu, lam - mu) for lam, nu2 in zip(lams, nu2s))
        sigma = sigma_sum / len(corpus)
        sigma_inv = np.linalg.inv(sigma)
        # sigma_inv = np.eye(len(mu))

        topic_count = len(mu)

        beta_ss = np.zeros((topic_count, vocab_size))
        for doc, phi in zip(corpus, phis):
            for i in range(topic_count):
                for n, word in enumerate(doc):
                    beta_ss[i, word] += phi[n, i]

        log_beta = np.zeros((topic_count, vocab_size))
        for i in range(topic_count):
            sum_term = sum(beta_ss[i])

            if sum_term == 0:
                sum_term = (-1000) * vocab_size
                print(sum_term)
            else:
                sum_term = np.log(sum_term)

            for j in range(vocab_size):
                log_beta[i, j] = np.log(beta_ss[i, j]) - sum_term
        # print (np.exp(log_beta))
        return mu, sigma_inv, log_beta


def main():
    text = []
    filenames = []
    for file in sorted((fname for fname in os.listdir('20newsgroups') if not fname.endswith('.csv')), key=int):
        if file.endswith('csv'):
            continue
        filenames.append(file)
        file = open(path.join('20newsgroups', file), 'r')
        text.append(file.read().split())

    words = sorted(set(sum(text, [])))
    documents = []
    for row in text:
        documents.append([words.index(word) for word in row])

    open('words.txt', 'w').write(' '.join(words))

    topic_count = 20
    mod = init_model_param(topic_count, len(words))
    # mu, sigma_inv, log_beta = mod

    after = sum(lhood_bnd(doc, self.init_variational_factor(topic_count, len(doc)), mod) for doc in documents)
    print('init ', after)
    for _ in range(1000):
        before = after
        corpus_var = expectation(documents, mod)
        mod = maximization(documents, corpus_var, len(words))

        after = sum(lhood_bnd(doc, var, mod) for doc, var in zip(documents, corpus_var))
        print('lhood = ', after)
        print(((before - after) / before))
        if ((before - after) / before) < 0.001:
            break
    mu, sigma_inv, log_beta = mod
    np.savetxt('beta.txt', np.exp(log_beta))
    corpus_lam = np.array([lam for zeta, phi, lam, nu2 in corpus_var])
    np.savetxt('corpus-lam.txt', corpus_lam)


if __name__ == '__main__':
    main()
