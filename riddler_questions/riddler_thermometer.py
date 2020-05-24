import numpy as np
#%%
def riddler_therm(a, b):
    func_output = round((5/9)*((10*a) + b - 32), 0) - 10*b - a
    print("Output of (a = {0} and b = {1}) = {2}".format(a, b, func_output))
    return func_output
#%%

# riddler_therm(test_a, test_b)

for a in range(0,10):
    for b in range(0, 10):
        mirror = riddler_therm(a, b)
        if mirror == 0:
            print("Solution is: a={0} and b={1}".format(a,b))
            break

        