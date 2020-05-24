# -*- coding: utf-8 -*-
"""
Created on Tue May 19 22:35:25 2020

@author: jopentak
"""

import numpy as np
#%%
N = 20
cards = np.arange(N, N+5)

def possible_n(cards_drawn):
    """Calculates possible values of n, given cards drawn"""
    upper_bound = np.min(cards_drawn)
    lower_bound = np.max(cards_drawn) - 4
    return np.arange(lower_bound, upper_bound+1)

def cards_remain(cards_drawn, n):
    """Given a set of drawn cards and n, returns a list of remaining cards"""
    return np.setdiff1d(np.arange(n, n+5), cards_drawn)
#%%

#%%
TEST_CASE = np.random.choice(cards, 2, False)
TEST_CASE = [20, 21]
poss_cards = possible_n(TEST_CASE)
print("Random card draw: " + str(TEST_CASE))
print("Test output: ")
print(poss_cards)
#%%
cards_left = cards_remain(TEST_CASE, 20)
print("Test remaining cards: ")
print(cards_left)