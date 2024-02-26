import json
import pickle

from gensim.corpora import Dictionary

from pygtm import GTM

model = GTM.load('model.pkl')

model.train()


