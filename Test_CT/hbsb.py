import numpy as np
from matplotlib import pyplot as plt

hb = np.load('batch.npy')
plt.imshow(hb[0,:,:],cmap=plt.get_cmap('gray'))
plt.show()
