import numpy as np 
import random as rd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import wordfreq
import random as rd
import json 

nlp = spacy.load('de_core_news_sm')


class GameState:
    def __init__(self): 
        self.V = self.get_wlist()[:10]
        self.TARGET = rd.choice(self.V)

        self.FREQ = {w: wordfreq.word_frequency(w, "de") for w in self.V}
        self.PLAYERS = ["P1", "P2", "P3"]
        self.ROLES = {"P1": "agent", "P2": "agent", "P3": "imposter"}
        self.TARGET = rd.sample(self.V, 1)[0]
        self.RATIONALITY = {"P1": 3, "P2": 3, "P3": 5}
        self.SIMS = self.get_sim_table(self.V, self.TARGET)
        self.W = []
        for p in self.PLAYERS:
            for w in self.V:
                self.W.append((p,w))


    def get_wlist(self, pos="NOUN", n=100, from_n = 10000, savepath="../output/random_nouns.json"):
        frequent_words = nlp(" ".join(wordfreq.top_n_list("de", from_n)))
        frequent_nouns = [w.lemma_ for w in frequent_words if (w.pos_==pos and w.lemma_[0].isupper())]
        rd_n = rd.sample(frequent_nouns, n)
        if savepath:
            with open(savepath, "w") as final:
                json.dump(rd_n, final)
        return(rd_n)
    
    # Load a large English model with word vectors included
    # Access the vector for a specific word
    def get_similarity(self, u, t): 
        u_vec = nlp(u).vector
        t_vec = nlp(t).vector
        return(cosine_similarity([u_vec], [t_vec])[0][0])

    def get_sim_table(self, V, t ): 
        # dim: m x n, so for each of the m target words there are similarities for each of the n words 
        sim_vec = []
        for t in V: 
            vec_t = []
            for w in V: 
                if t==w: 
                    vec_t.append(0)
                vec_t.append(self.get_similarity(w,t))
            sim_vec.append(vec_t)
        return(np.array(sim_vec))
    

gs = GameState()