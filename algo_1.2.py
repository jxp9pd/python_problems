import numpy as np

#%%
def compare_to(a, b):
    if a < b:
        return -1
    elif a == b:
        return 0
    return 1

def insertion_sort(a_list):
    '''My initial approach for insertion sort'''
    print('Original list: ')
    print(a_list)
    #If list already sorted, return it.
    if len(a_list) <= 1:
        return a_list
    curr_point = 1
    for i in range(len(a_list)):
        while a_list[curr_point] < a_list[curr_point - 1] and curr_point > 0:
            temp = a_list[curr_point]
            a_list[curr_point] = a_list[curr_point - 1]
            a_list[curr_point - 1] = temp
            curr_point -= 1
        curr_point = i+1
    print('Sorted List: ')
    print(a_list)
    return a_list
#%%
def book_insertion(A):
    '''Insertion sort done slightly nicer'''
    print('Original list: ')
    print(A)
    for j in range (1, len(A)):
        key = A[j]
        i = j-1
        while i>=0 and A[i]<key:
            A[i+1] = A[i]
            i -= 1
        A[i+1] = key
    print('Sorted List: ')
    print(A)
    return A
#%%
def bin_addition(A,B):
    '''Binary addition of two n-sized arrays'''
    carry = 0
    C = []
    for i in range(len(A)-1, -1, -1):
        digit = A[i] + B[i] + carry
        if digit == 0:
            C.append(0)
        elif digit == 1:
            C.append(0)
            carry = 0
        elif digit == 2:
            C.append(0)
            carry = 1
        else:
            C.append(1)
            carry = 1
    C.append(carry)
    return C[::-1]
#%%
        
#%%
if __name__ == '__main__':
    '''Test Case 1: Insertion Sort'''
    #test = np.random.choice(20, 25)
    #insertion_sort(test)
    #book_insertion(test)
    '''Test Case 2: Binary Addition'''
    A = [1,0,1]
    B = [0,1,1]
    C = bin_addition (A, B)
