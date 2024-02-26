import json
import pickle

import numpy as np
from gensim.corpora import Dictionary
from numba import typed, types
from scipy.spatial.distance import squareform, pdist

from pygtm import GTM


def main():
    # Load the document
    documents = json.load(open('suzhou_sense_for_gtm.json', 'r', encoding='UTF-8'))

    # Create a dictionary from the documents
    dictionary = Dictionary(documents)
    dictionary.save_as_text('dictionary.txt')

    # Create a corpus from the documents
    corpus = [dictionary.doc2idx(doc) for doc in documents]

    # Save the corpus
    with open('corpus.jsonl', 'w', encoding='utf8') as f:
        for doc in corpus:
            f.write(json.dumps(doc) + '\n')

    # Generate simulated location
    location_count = 5

    # Generate a random location for each document
    locations = np.random.randint(0, location_count, len(documents))

    # Save the location
    with open('locations.pkl', 'wb') as f:
        pickle.dump(locations, f)

    # Generate a random coordinate for each location
    coords = np.random.uniform(0, 1, (location_count, 2))

    # Calculate the distance matrix using the coordinates and pdist
    distance_matrix = squareform(pdist(coords))

    # Calculate the weight matrix using the distance matrix and gaussian kernel
    weight_matrix = np.exp(-distance_matrix ** 2)

    # Initialize the GTM model
    gtm = GTM(
        30, len(dictionary), location_count, np.float64(weight_matrix),
        variational_rate=1e-1, em_rate=1e-1
    )

    corpus_input = typed.List.empty_list(types.int32[::1])
    for doc in corpus:
        corpus_input.append(np.array(doc))

    # Train the GTM model
    gtm.train(corpus_input, locations)

    gtm.save('model.pkl')

    print()


if __name__ == '__main__':
    main()
