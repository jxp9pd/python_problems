'''
Sophia Pentakalos
John Pentakalos

Check Permutation: Given two strings, write a method a method to decide if one is a permutation of the other.
'''

def is_Permutation(str1, str2):
    return sorted(str1) == sorted(str2)

def is_permute(str1, str2):
    if len(str1) != len(str2): return False
    my_dict = {}
    for char in str1:
        if char not in my_dict: my_dict[char] = 1
        else: my_dict[char] += 1
    print(my_dict)
    for char in str2:
        if char not in my_dict: return False
        if my_dict[char] == 0: return False
        my_dict[char] -= 1
    return True
    
#%%
is_permute('abab', 'baba')
