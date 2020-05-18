"""Riddler Dungeons and Dragons"""

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 20)
#%%
#Computing EV for Advantage of disadvantage
dice_rolls = np.arange(1, 21)
arr_1 = np.repeat(dice_rolls, 20).reshape(-1, 1)
arr_2 = np.tile(dice_rolls, 20).reshape(-1, 1)
dice_combos = np.hstack((arr_1, arr_2))
#%%
#Produces a table of all possible combinations
dice_df = pd.DataFrame(dice_combos, columns=["Dice1", "Dice2"])
#Produces the disadvantage and advantage outcomes of each dice roll combo
dice_df["DisadvantageRoll"] = dice_df.apply(lambda x: np.min([x["Dice1"],
                                                   x["Dice2"]]), axis=1)
dice_df["AdvantageRoll"] = dice_df.apply(lambda x: np.max([x["Dice1"],
                                                x["Dice2"]]), axis=1)
#%%
#Calculates the probability of each combo originating from both a disadvantage
#roll or an advantage roll
disadvantage_pmf = dice_df.DisadvantageRoll.value_counts()/400
advantage_pmf = dice_df.AdvantageRoll.value_counts()/400

def prob_roll_disadvantage(a, b):
    """Probability of a roll originating from two disadvantage dice outcomes"""
    return disadvantage_pmf[a] * disadvantage_pmf[b]

def prob_roll_advantage(a, b):
    """Probability of a roll originating from two disadvantage dice outcomes"""
    return advantage_pmf[a] * advantage_pmf[b]
#%%
dice_df["DisadvantageProb"] = dice_df.apply(lambda x: prob_roll_disadvantage(x["Dice1"], x["Dice2"]), axis=1)
dice_df["AdvantageProb"] = dice_df.apply(lambda x: prob_roll_advantage(x["Dice1"], x["Dice2"]), axis=1)
#Sanity check
print("Sum probability of events in disprob: {}".format(np.sum(dice_df["DisadvantageProb"])))
print("Sum probability of events in adprob: {}".format(np.sum(dice_df["AdvantageProb"])))
#%%
#Column for advantage of disadvantage and disadvantage of advantage
dice_df["AdofDis"] = dice_df.apply(lambda x: x["AdvantageRoll"]*x["DisadvantageProb"], axis=1)
dice_df["DisofAd"] = dice_df.apply(lambda x: x["DisadvantageRoll"]*x["AdvantageProb"], axis=1)
dice_df

print("EV roll of advantage of disadvantage is {}".format(np.sum(dice_df["AdofDis"])))
print("EV roll of disadvantage of advantage is {}".format(np.sum(dice_df["DisofAd"])))
