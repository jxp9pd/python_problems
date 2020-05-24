"""
Created on Sat May 23 23:41:59 2020
https://fivethirtyeight.com/features/somethings-fishy-in-the-state-of-the-riddler/
"""
import pandas as pd
import numpy as np

DOWNLOADS = "Downloads/"
#%%
#Read in the word list and list of states
word_list = pd.read_csv(DOWNLOADS + "word.list", header=None)
states = pd.read_csv(DOWNLOADS + "states.txt", sep='\n', header=None)
word_list.columns = ["words"]
word_list["words"] = word_list["words"].astype('string')
word_list.dropna(inplace=True)
states.columns = ["States"]
#%%
#Add a column for the unique characters in each word
states["States"] = states["States"].str.replace(" ", "").str.lower()
states['Unique_chars'] = states.apply(lambda x: ''.join(sorted(set(x['States']))), axis=1)
# word_list.drop_duplicates(subset="Unique_chars", inplace=True)
#%%
#Creating a map of which states to eliminate for a seen character
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
STATE_MATCHER = {}
for letter in ALPHABET:
    STATE_MATCHER[letter] = set(states[states["Unique_chars"].str.contains(
        letter)].index.values)
#%%
#Sort the words so longest words come first
test_words = word_list.sample(frac=0.1, replace=False, random_state=1)
test_words['length'] = test_words['words'].str.len()
test_words.sort_values('length', ascending=False, inplace=True)
#%%
word_list['length'] = word_list['words'].str.len()
word_list.sort_values('length', ascending=False, inplace=True)
#%%
#Find the mackerel('s)!
mackerel_length = 0
mackerels = []
state_names = []
for index, row in word_list.iterrows():
    #Ignore words that aren't long enough for consideration
    if len(row["words"]) < mackerel_length:
        print("Found all the mackerels!")
        break
    else:
        word = row["words"]
        unique_chars = ''.join(set(word))
        poss_states = set(np.arange(0, 50))
        for char in unique_chars:
            poss_states = poss_states - STATE_MATCHER[char]
            # if you've already eliminated all states, just move on.
            # if len(poss_states) == 0:
            #     continue
        if len(poss_states) == 1:
            mackerel_length = len(word)
            mackerels.append(word)
            state_names.append(poss_states.pop())
mackerel_states = [states["States"].iloc[i] for i in state_names]
#%%
print("Found the following mackerel words:")
for word, state in zip(mackerels, mackerel_states):
    print("Mackerel word was {} for the state of {}".format(word, state))
print("Longest word was {} letters long.".format(len(mackerels[0])))
