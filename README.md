# imposter_RSA

A git repository for my implementation of the game imposter using Rational Speech Act theory. This is done to model language use in an adverserial setting, comparable to uses of coded language like dog whistles. 
The repository contains the following files:

In the folder *notebooks*:     - 
- The python script *gamestate.py* contains the "GameState" class containing all information that is available to all players (like vocabulary, players) and the "SpeakerAgent" class which contains the actual modelling of the speaker 
- The IPython notebook *run_experiments.ipynb* calls an object of the speaker class and uses it to run the experiments. It also generates the figures used in the term paper. 

In the folder output: 
- the file *exp_sims.npy* contains the similarity matrix used in the experiments 
- the file *random_nouns.json* contains the random nouns sampled from the "wordfreq" python module (not used in the paper)
- the folders *e1*, *e2*, *e3* contain the speaker and listener probabilities as well as the figures from each experiment 