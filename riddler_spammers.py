"""https://fivethirtyeight.com/features/can-you-catch-the-free-t-shirt/"""
import numpy as np

#%%
def sample_comments(days, avg_rate=1):
    """Completes one random sample of how many comments are produced"""
    start_posts = 1
    for i in range(days):
        new_posts = np.sum(np.random.poisson(avg_rate, start_posts))
        start_posts += new_posts
    return start_posts - 1
#%%
one_iteration = sample_comments(3)
print("One run output of spam post count: {0}".format(one_iteration))
#%%
#Monte Carlo Simulation for spammers
num_iterations = 10000
results = []
for i in range(num_iterations):
    results.append(sample_comments(3))
results = np.array(results)
print("Expected number of spam posts after 3 days is {}".format(np.mean(results)))
