'''
Sophia Pentakalos, John Pentakalos
'''
#1. Create the following list
#[0, 9, 10, 19, 20, 29, 30, 39, 40, 49, 50, 59, 60, 69, 70, 79, 80, 89, 90, 99]
#John Solution
jp = [i*5 if i%2==0 else (i+1)*5-1 for i in range(20)]
sp = [int(x) for x in range(100) if x % 10 == 0 or x%10 == 9]

#%%

