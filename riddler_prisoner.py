"""https://fivethirtyeight.com/features/can-you-flip-your-way-to-freedom/"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

#%%
def prob_freedom(prob_flip):
    """Calculates probability of freedom for a particular prob_flip"""
    n_flip = 1-prob_flip
    p_success = 0.5*comb(4, 1)*np.power(n_flip, 3)*prob_flip + \
               0.25*comb(4, 2)*np.power(n_flip, 2)*np.power(prob_flip, 2) + \
               (1/8)*comb(4, 3)*n_flip*np.power(prob_flip, 3) + \
               (1/16)*np.power(prob_flip, 4)
    return p_success
#%%
def plot_freedom(tries, freedom):
    """Simple line plot of freedom respective to how often we flip"""
    plt.plot(tries, freedom)
    plt.title("Probability of freedom based on how likely a prisoner is to flip")
    plt.xlabel("Probability of prisoner flip")
    plt.ylabel("Probability of release")
    plt.show()
#%%
v_prob = np.vectorize(prob_freedom)
possible_probs = np.arange(0, 1, 0.005)
freedom_probs = v_prob(possible_probs)
best_pct = possible_probs[np.argmax(freedom_probs)]*100
print("Prisoners should flip a coin {0}% of the time, earning freedom {1:.2f}% of the time.".format(
        best_pct, np.max(freedom_probs)*100))
plot_freedom(possible_probs, freedom_probs)
