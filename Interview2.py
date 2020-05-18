'''
Sophia Pentakalos
John Pentakalos

Check Permutation: Given two strings, write a method to decide if one is
a permutation of the other.
'''

def is_permutation(str1, str2):
    '''Sophia sorting permutation solution'''
    return sorted(str1) == sorted(str2)

def is_permute(str1, str2):
    '''John dictionary solution'''
    if len(str1) != len(str2):
        return False
    my_dict = {}
    for char in str1:
        if char not in my_dict:
            my_dict[char] = 1
        else:
            my_dict[char] += 1
    for char in str2:
        if char not in my_dict or my_dict[char] == 0:
            return False
        my_dict[char] -= 1
    return True
is_permute('abab', 'baba')
#%%
import pdb
def moveZeroes(nums):
    """
    Do not return anything, modify nums in-place instead.
    """
    zeros = 0
    for i in range(len(nums)):
        if nums[i-zeros] == 0:
            nums.pop(i-zeros)
            nums.append(0)
            zeros += 1
#%%
test_nums = [0, 1, 0, 3, 12]
moveZeroes(test_nums)
print(test_nums)
#%%
def bounce(n):
    if n >= 0:
        bounce(n - 1)
        print(n)
        if n:
            print(n)
bounce(4)
#%%
def findDuplicate(nums):
    pdb.set_trace()
    i = nums[0]
    j = nums[nums[0]]
    while (nums[i] != nums[j]) and (i != j):
        i = nums[i]
        j = nums[nums[j]]
    return nums[i]
#%%
test_1 = [1,3,4,2,2]
print(findDuplicate(test_1))

test_2 = [2,5,9,6,9,3,8,9,7,1]
print(findDuplicate(test_2))
#%%
def sortColors(nums):
    """
    Do not return anything, modify nums in-place instead.
    """
    counter = [0,0,0]
    for n in nums:
        counter[n] += 1
    i = 0 # Keeps track of which color to be filled
    j = 0 # Keeps track of where to put the num.
    while sum(counter) > 0:
        if counter[i] == 0:
            i += 1
        else:
            nums[j] = i
            counter[i] -= 1
            j += 1
#%%
test_1 = [2,0,2,1,1,0]
sortColors(test_1)
    

