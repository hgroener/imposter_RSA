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
    def __init__(self, sim_file="../output/sims.npy", vsize=10, dummy=False, dummy_vocab=["Katze", "Pferd", "Stall", "Dinosaurier", "Hund"], target=None
): 

        if dummy:
            # uses preset vocabulary instead of sampling from highest frequency nouns 
            self.V = dummy_vocab
        else:
            # samples vocabulary of given size from highest frequency nouns 
            self.V = self.get_wlist(n=vsize)
        #print("V: ", self.V)
        if target: 
            # uses preset target
            self.TARGET=target
        else:
            # uses random target
            self.TARGET = rd.choice(self.V)
        print(f"randomly chosen target word: {self.TARGET}")
        # gets word frequencies of each word in vocabulary 
        self.FREQ = {w: wordfreq.word_frequency(w, "de") for w in self.V}
        freq_sum = sum(self.FREQ.values())
        if dummy:
            # uses uniform distribution for imposter's target word priors 
            self.FREQ_NORMED = {w: 1/len(self.V) for w in self.V}
        else:
            # uses normalized distirbution of word frequency values for target word priors 
            self.FREQ_NORMED = {w: f/freq_sum for w,f in self.FREQ.items()}
        self.PLAYERS = ["P1", "P2", "P3"]

        self.ROLES = {"P1": "agent", "P2": "agent", "P3": "imposter"}
        
        if sim_file and path.isfile(sim_file):
            sims = np.load(sim_file)
            print(f"similarity table loaded from {sim_file}")
        else:
            # calculates matrix of word similarities, saves them to file 
            sims = self.get_sim_table(savefile=sim_file)
        
        # each player uses their own similarity matrix, which can be manipulated
        self.SIMS = {p: sims.copy() for p in self.PLAYERS}


    # gets list of highest frequency nouns from wordfreq module, saves them to .json file 
    def get_wlist(self, pos="NOUN", n=100, from_n = 1000, savepath="../output/random_nouns.json"):

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

    # get similarity matrix of each word in vocabulary, save to file 
    def get_sim_table(self, savefile="../output/sims.npy"):
        # dim: m x n, so for each of the m target words there are similarities for each of the n words 
        sim_vec = []
        for t in self.V: 
            vec_t = []
            for w in self.V: 
                # normalize cosine similarity to get it in range [0,1]
                vec_t.append(self.get_similarity(w,t))
            sim_vec.append(vec_t)
        sim_vec=np.array(sim_vec)
        sim_vec = sim_vec - sim_vec.min()
        for n in range(len(self.V)):
            sim_vec[n,n] = 0
        print(f"similarity table of size {len(self.V)}x{len(self.V)} generated")
        if savefile:
            np.save(savefile, sim_vec)
            print(f"similarity table saved to {savefile}")
        return(sim_vec)



class Speaker_Agent:
    def __init__(self, player, gamestate, logger=None, i_weight = 2, t_weight=5, alpha=1): 
        self.player = player
        self.gs = gamestate
        # sets logger to save probablity distributions 
        self.logger = logger
        # rationality value 
        self.alpha = alpha 
        # gets priors for each player given each role 
        self.priors = self.get_priors_per_player()
        # sets weights for each part of the utility function 
        self.i_weight = i_weight
        self.t_weight = t_weight

    def get_priors_per_player(self): 
        priors = {p: {} for p in self.gs.PLAYERS}
        for p in self.gs.PLAYERS: 

            for role in ["imposter", "agent"]: 
                # imposter knows identity of imposter but not of target word
                if role=="imposter":
                    ki = True
                    kt = False
                # agent knows identity of target word but not of imposter
                else:
                    ki = False
                    kt = True

                # gets priors according to role
                i_priors, t_priors = self.get_priors(p, knows_t=kt, knows_imp=ki)
                priors[p][role]={}
                priors[p][role]["i"]=i_priors
                priors[p][role]["t"]=t_priors
        return(priors)



    def get_priors(self, player, knows_t = True, knows_imp = False):
        i_priors = []
        t_priors = []
        # P1 weiß nicht wer der Imposter ist 
        for p in self.gs.PLAYERS: 
            if knows_imp: 
                if p==player: 
                    i_priors.append(1)
                else:
                    i_priors.append(0)
            else:
                if p==player:
                    i_priors.append(0)
                else: 
                    # agent knows that they themself are not the agent, uniform distribution over other players
                    p_num = len(self.gs.PLAYERS)
                    i_priors.append(1/(p_num-1))
        
        for w in self.gs.V: 
            if knows_t: 
                if w==self.gs.TARGET:
                    t_priors.append(1)
                else:
                    t_priors.append(0)
            else:
                # uniform distribution if using preset vocabulary, else normalized word frequencies 
                t_priors.append(self.gs.FREQ_NORMED[w])
                                
                    
        i_priors = np.array(i_priors)
        t_priors = np.array(t_priors)

        return((i_priors, t_priors))
    

    # calculates probabilities of imposter identities given utterance u and world w
    def get_P_u_w(self,u,w,speaker="P1", listener="P2"):
        imp_ind, t_ind = w

        if self.gs.PLAYERS[imp_ind]==speaker: 
            P_u_w = sum([self.priors[speaker]["imposter"]["t"][n] * self.gs.SIMS[listener][n, u] for n in range(len(self.gs.V))])
        else:
            P_u_w = self.gs.SIMS[listener][t_ind, u]
            
        return(P_u_w)

    # listener function for agents, calculates probabilities of imposter identity given utterance, target word 
    def L0_a(self, u, w, listener="P2", speaker="P1"):
        #logger.info(f"calculating P({w}|{u}) for L0...")
        imp, t = w
        imp = self.gs.PLAYERS.index(imp)
        t = self.gs.V.index(t)        
        u = self.gs.V.index(u)
        priors_l = self.priors[listener]

        #logger.info(f"P(u|w) in L(0): {self.get_P_u_w(u,(imp,t),t_imp) }, priors: {priors_l[imp][t]}")
        P_w_u = self.get_P_u_w(u,(imp,t), speaker=speaker, listener=listener) * priors_l["agent"]["i"][imp]
        #logger.info(f"P(w|u) in L(0): {P_w_u}")
        SUM = np.nansum([self.get_P_u_w(u,(i_n, t), speaker=speaker, listener=listener) * priors_l["agent"]["i"][i_n] for i_n in range(len(self.gs.PLAYERS))])
        #logger.info(f"SUM in L0: {SUM}")
        # prevent dividing by 0 
        if SUM==0:
            P_w_u = 0
        else:    
            P_w_u = P_w_u/SUM
            
        #logger.info(f"normalized P(w|u) in L(0): {P_w_u}")

        return(P_w_u)

    # calculates probabilities of target word identities given utterance u
    def get_P_u_t(self,u,t, listener="P3"): 

        P_u_t = self.gs.SIMS[listener][t, u]

        return(P_u_t)

    # listener function for imposter, calculates probabilities of target word identity given utterance, imposter identity  
    def L0_t(self, u, t, listener="P3"): 
        # literal listener function for the imposter
        t = self.gs.V.index(t)
        u = self.gs.V.index(u)
        priors_l = self.priors[listener]
        P_t_u = self.get_P_u_t(u,t, listener=listener) * priors_l["imposter"]["t"][t]
        SUM = np.nansum([self.get_P_u_t(u,t_n, listener=listener) * priors_l["imposter"]["t"][t_n] for t_n in range(len(self.gs.V))])
        # prevent dividing by 0 
        if SUM==0:
            P_t_u = 0
        else:    
            P_t_u = P_t_u/SUM
        return(P_t_u)

    # utility function of pragmatic speaker, calculates utility of utterances given, target word, imposter identity
    def utility(self, u, i, t, speaker="P1", adversarial=True):
        partner = [p for p in self.gs.PLAYERS if not p in [speaker, i]][0]
        l0_a = self.L0_a(u,(i,t),listener=partner, speaker=speaker)
        U_a = l0_a
        if adversarial: 
            l0_i = self.L0_t(u,t,listener=i)
            U_i = l0_i
            # weighted difference between utilities for other agent, imposter
            U = self.i_weight * U_a - self.t_weight * U_i
            if self.logger: 
                self.logger.info(f"outputs of L0 functions within utility function: L0(t|u): {l0_i}, L0(i|u): {l0_a}")
                self.logger.info(f"imposter component of utility function: {U_i}\nagent component of utility function: {U_a}")
        else:
            # calculates utility without trying to minimize information gain for imposter
            U = U_a
            if self.logger: 
                self.logger.info(f"output of L0 function within utility function: L0(i|u): {l0_a}")

        return(U)

    # pragmatic speaker function without uncertainty about imposter identity 
    def S1_i_t(self, u, i, t, speaker="P1", adversarial=True):
        P_u = np.exp(self.alpha*self.utility(u,i,t, speaker=speaker, adversarial=adversarial))
        SUM = np.nansum([np.exp(self.alpha*self.utility(u_n, i, t, speaker=speaker, adversarial=adversarial)) for u_n in self.gs.V])
        # prevent dividing by 0 
        if SUM==0:
            P_u = 0
        else:
            P_u = P_u/SUM

        return(P_u)

    # pragmatic speaker function with uncertainty about imposter identity 
    def S1_certain(self, u, t, speaker="P1",adversarial=True): 

        # gets weighted sum of output probabilites given each possible imposter identity 
        def get_i_sum(u,t,speaker="P1"):
            P_u = [self.priors[speaker]["agent"]["i"][self.gs.PLAYERS.index(i)] * self.S1_i_t(u,i,t, speaker=speaker, adversarial=adversarial) for i in self.gs.PLAYERS if i!=self.player]
            if self.logger: 
                for n, p in enumerate([player for player in self.gs.PLAYERS if player!=self.player]):
                    self.logger.info(f"S1(u|{p},{self.gs.TARGET})={P_u[n]}")
            P_u_sum = np.nansum(P_u)
            return(P_u_sum)

        P_u = get_i_sum(u,t,speaker=speaker)
        SUM = np.nansum([get_i_sum(u_n,t,speaker=speaker) for u_n in self.gs.V])
        # prevent dividing by 0 
        if SUM == 0: 
            P_u = 0 
        else:
            P_u = P_u/SUM
        return(P_u)

