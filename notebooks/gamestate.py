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
            self.V = dummy_vocab
        else:
            self.V = self.get_wlist(n=vsize)
        #print("V: ", self.V)
        if target: 
            self.TARGET=target
        else:
            self.TARGET = rd.choice(self.V)
        print(f"randomly chosen target word: {self.TARGET}")
        self.FREQ = {w: wordfreq.word_frequency(w, "de") for w in self.V}
        freq_sum = sum(self.FREQ.values())
        if dummy:
            self.FREQ_NORMED = {w: 1/len(self.V) for w in self.V}

        else:
            self.FREQ_NORMED = {w: f/freq_sum for w,f in self.FREQ.items()}
        self.PLAYERS = ["P1", "P2", "P3"]

        self.ROLES = {"P1": "agent", "P2": "agent", "P3": "imposter"}
        self.RATIONALITY = {"P1": 3, "P2": 3, "P3": 3}

        if sim_file and path.isfile(sim_file):
            sims = np.load(sim_file)
            print(f"similarity table loaded from {sim_file}")
        else:
            sims = self.get_sim_table(self.V, self.TARGET, savefile=sim_file)
        self.SIMS = {p: sims for p in self.PLAYERS}
        self.W = []
        for p in self.PLAYERS:
            for w in self.V:
                self.W.append((p,w))


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



class Speaker_Agent:
    def __init__(self, player, gamestate, logger=None, i_weight = 0.6): 
        self.player = player
        self.gs = gamestate
        self.logger = logger
        self.role = self.gs.ROLES[player]
        self.alpha = self.gs.RATIONALITY[player]
        self.priors = self.get_priors_per_player()
        self.i_weight = i_weight

    def get_priors_per_player(self): 
        priors = {p: {} for p in self.gs.PLAYERS}
        for p in self.gs.PLAYERS: 

            for role in ["imposter", "agent"]: 
                if role=="imposter":
                    ki = True
                    kt = False
                else:
                    ki = False
                    kt = True

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
                    p_num = len(self.gs.PLAYERS)
                    i_priors.append(1/(p_num-1))
        
        for w in self.gs.V: 
            if knows_t: 
                if w==self.gs.TARGET:
                    t_priors.append(1)
                else:
                    t_priors.append(0)
            else:
                t_priors.append(self.gs.FREQ_NORMED[w])
                                
                    
        i_priors = np.array(i_priors)
        t_priors = np.array(t_priors)

        return((i_priors, t_priors))
    

    def get_P_u_w(self,u,w,speaker="P1", listener="P2"):
        imp_ind, t_ind = w
        # sollte so passen, weil wenn L=imp sind Berechnungen egal, in denen nicht S = imp
        # kommt mir sehr sus vor - warum wird einmal die ähnlichkeit mit dem in der welt enthalteten target berechnet und einmal mit dem target von imp?
        if self.gs.PLAYERS[imp_ind]==speaker: 
            P_u_w = sum([self.priors[speaker]["imposter"]["t"][n] * self.gs.SIMS[listener][n, u] for n in range(len(self.gs.V))])
        else:
            P_u_w = self.gs.SIMS[listener][t_ind, u]
            
        return(P_u_w)

    def L0_a(self, u, w, listener="P2", speaker="P1"):
        # literal listener function for agents 
        #logger.info(f"calculating P({w}|{u}) for L0...")
        imp, t = w
        imp = self.gs.PLAYERS.index(imp)
        t = self.gs.V.index(t)        
        u = self.gs.V.index(u)
        priors_l = self.priors[listener]

        #logger.info(f"P(u|w) in L(0): {self.get_P_u_w(u,(imp,t),t_imp) }, priors: {priors_l[imp][t]}")
        P_w_u = self.get_P_u_w(u,(imp,t), speaker=speaker, listener=listener) * priors_l["agent"]["i"][imp]
        #logger.info(f"P(w|u) in L(0): {P_w_u}")
        SUM = np.nansum([self.get_P_u_w(u,(i_n, t), speaker=speaker) * priors_l["agent"]["i"][i_n] for i_n in range(len(self.gs.PLAYERS))])
        #logger.info(f"SUM in L0: {SUM}")
        if SUM==0:
            P_w_u = 0
        else:    
            P_w_u = P_w_u/SUM
            
        #logger.info(f"normalized P(w|u) in L(0): {P_w_u}")

        return(P_w_u)


    
    def get_P_u_t(self,u,t, listener="P3"): 

        P_u_t = self.gs.SIMS[listener][t, u]

        return(P_u_t)

    
    def L0_t(self, u, t, listener="P3"): 
        # literal listener function for the imposter
        t = self.gs.V.index(t)
        u = self.gs.V.index(u)
        priors_l = self.priors[listener]
        P_t_u = self.get_P_u_t(u,t, listener=listener) * priors_l["imposter"]["t"][t]
        SUM = np.nansum([self.get_P_u_t(u,t_n, listener=listener) * priors_l["imposter"]["t"][t_n] for t_n in range(len(self.gs.V))])
        if SUM==0:
            P_t_u = 0
        else:    
            P_t_u = P_t_u/SUM
        return(P_t_u)
    
    def utility(self, u, i, t, speaker="P1", adversarial=True):
        partner = [p for p in self.gs.PLAYERS if not p in [speaker, i]][0]
        l0_a = self.L0_a(u,(i,t),listener=partner, speaker=speaker)
        U_a = l0_a#np.log(l0_i)
        if adversarial: 
            l0_i = self.L0_t(u,t,listener=i)
            U_i = l0_i
            # U_i is probably weighed to highly like this 
            U = U_a + self.i_weight * (1-U_i)
            if self.logger: 
                self.logger.info(f"outputs of L0 functions within utility function: L0(t|u): {l0_i}, L0(i|u): {l0_a}")
                self.logger.info(f"imposter component of utility function: {U_i}\nagent component of utility function: {U_a}")
        else:
            U = U_a
            if self.logger: 
                self.logger.info(f"output of L0 function within utility function: L0(i|u): {l0_a}")

        return(U)
    
    def S1_i_t(self, u, i, t, alpha=1, speaker="P1",  adversarial=True):
        # gets other player that is not imposter
        partner = [p for p in self.gs.PLAYERS if not p in [speaker,i]][0]
        #P_u = np.exp(alpha*self.utility(u,i,t, partner=partner, speaker=speaker, log=True, i_weight=i_weight))
        #P_u = np.exp(alpha*self.utility(u,i,t, speaker=speaker, log=True, i_weight=i_weight, adversarial=adversarial))
        P_u = alpha*self.utility(u,i,t, speaker=speaker, adversarial=adversarial)
        #SUM = np.nansum([np.exp(alpha*self.utility(u_n, i, t, speaker=speaker, adversarial=adversarial)) for u_n in gs.V])
        SUM = np.nansum([alpha*self.utility(u_n, i, t, speaker=speaker, adversarial=adversarial) for u_n in self.gs.V])
        if SUM==0:
            P_u = 0
        else:
            P_u = P_u/SUM

        return(P_u)
    
    def S1_certain(self, u, t, speaker="P1", alpha=1,adversarial=True): 
        
        def get_i_sum(u,t,speaker="P1", alpha=1):
            P_u = [self.priors[speaker]["agent"]["i"][self.gs.PLAYERS.index(i)] * self.S1_i_t(u,i,t, alpha=alpha, speaker=speaker, adversarial=adversarial) for i in self.gs.PLAYERS if i!=self.player]
            if self.logger: 
                for n, p in enumerate([player for player in self.gs.PLAYERS if player!=self.player]):
                    self.logger.info(f"S1(u|{p},{self.gs.TARGET})={P_u[n]}")
            P_u_sum = np.nansum(P_u)
            return(P_u_sum)

        P_u = get_i_sum(u,t,speaker=speaker, alpha=alpha)
        SUM = np.nansum([get_i_sum(u_n,t,speaker=speaker, alpha=alpha) for u_n in self.gs.V])
        if SUM == 0: 
            P_u = 0 
        else:
            P_u = P_u/SUM
        return(P_u)

