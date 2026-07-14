import numpy as np 
import random as rd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import wordfreq
import random as rd
import json 
import itertools
from os import path 

nlp = spacy.load('de_core_news_md')


class GameState:
    def __init__(self, sim_file="../output/sims.npy", vsize=10): 
                 #sim_file="/home/user/Documents/Sprachgebrauchsmodellierung_RSA/Hausarbeit_RSA/repo/output/sims.npy"): 
        self.V = self.get_wlist(n=vsize)
        #print("V: ", self.V)
        self.TARGET = rd.choice(self.V)
        print(f"randomly chosen target word: {self.TARGET}")
        self.FREQ = {w: wordfreq.word_frequency(w, "de") for w in self.V}
        freq_sum = sum(self.FREQ.values())
        self.FREQ_NORMED = {w: f/freq_sum for w,f in self.FREQ.items()}
        self.PLAYERS = ["P1", "P2", "P3"]
        play_order = self.PLAYERS.copy()
        rd.shuffle(play_order)
        print("randomly chosen play order:", play_order)
        self.PLAY_ORDER = itertools.cycle(play_order)
        self.ROLES = {"P1": "agent", "P2": "agent", "P3": "imposter"}
        self.RATIONALITY = {"P1": 3, "P2": 3, "P3": 5}
        self.HISTORY = {p: [] for p in self.PLAYERS}
        if sim_file and path.isfile(sim_file):
            self.SIMS = np.load(sim_file)
            print(f"similarity table loaded from {sim_file}")
        else:
            self.SIMS = self.get_sim_table(self.V, self.TARGET)
        self.W = []
        for p in self.PLAYERS:
            for w in self.V:
                self.W.append((p,w))


    def get_wlist(self, pos="NOUN", n=100, from_n = 1000, savepath="../output/random_nouns.json"):
                  #savepath="/home/user/Documents/Sprachgebrauchsmodellierung_RSA/Hausarbeit_RSA/repo/output/random_nouns.json"):
        if savepath and path.isfile(savepath): 
            with open(savepath) as f:
                rd_n = json.load(f)
                print(f"Vocabulary loaded from {savepath}")
        else:
            frequent_words = nlp(" ".join(wordfreq.top_n_list("de", from_n)))
            frequent_nouns = [w.lemma_ for w in frequent_words if (w.pos_==pos and w.lemma_[0].isupper())]
            rd_n = rd.sample(frequent_nouns, n)
            print(f"Vocabulary of size {n} sampled from {from_n} most frequent nouns.")
            if savepath:
                with open(savepath, "w") as final:
                    json.dump(rd_n, final)
                    print(f"Vocabulary saved to {savepath}")
        return(rd_n)
    
    # Load a large English model with word vectors included
    # Access the vector for a specific word
    def get_similarity(self, u, t): 
        u = nlp(u)
        t = nlp(t)
        return(u.similarity(t))

    def get_sim_table(self, V, t, savefile="../output/sims.npy"):
                      #savefile="/home/user/Documents/Sprachgebrauchsmodellierung_RSA/Hausarbeit_RSA/repo/output/sims.npy"): 
        # dim: m x n, so for each of the m target words there are similarities for each of the n words 
        sim_vec = []
        for t in V: 
            vec_t = []
            for w in V: 
                # normalize cosine similarity to get it in range [0,1]
                #vec_t.append((1 + self.get_similarity(w,t))/2)
                vec_t.append(self.get_similarity(w,t))
            sim_vec.append(vec_t)
        sim_vec=np.array(sim_vec)
        sim_vec = sim_vec - sim_vec.min()
        for n in range(len(V)):
            sim_vec[n,n] = 0
        print(f"similarity table of size {len(self.V)}x{len(self.V)} generated")
        if savefile:
            np.save(savefile, sim_vec)
            print(f"similarity table saved to {savefile}")
        return(sim_vec)
    

gs = GameState()
#print(gs.V)