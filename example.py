# %%
from deep_insight_face.datagen import generator
from matplotlib import pyplot as plt

gen = generator.triplet_datagenerator('lfw_data', 'lfw_data/pairs.txt')
fig = plt.figure()
for i in range(10):
    x1, y1 = next(gen)
    plt.imshow(x1)


# %%

gen = generator.facematch_datagenerator('lfw_data', 'lfw_data/pairs.txt')
fig = plt.figure()
for i in range(10):
    x1, y1 = next(gen)
    plt.imshow(x1)


# %%
# TODO: 
# - Verify generator script -- Doing
# - Verify training script and eval script -- Doing