import numpy as np
import matplotlib.pyplot as plt

dataset_name = 'web'
gan_type = 'CGAN'
nr = '1'
mmd_path = "runs/{}/{}/{}/checkpoint/mmd.npy".format(dataset_name, gan_type, str(nr))
mmd = np.load(mmd_path)
save_path = "evaluation/{}/{}/{}/mmd.jpg".format(dataset_name, gan_type, str(nr))
# mmd_path_2 = "runs/{}/{}/{}/checkpoint/mmd.npy".format(dataset_name, gan_type, str(nr_2))
# mmd_2 = np.load(mmd_path_2)

fig, axes = plt.subplots(1, 1, figsize=(12, 4))
axes.plot(np.arange(0, len(mmd)), mmd, color='green')
# axes.plot(np.arange(0, len(mmd_2)), mmd_2, color='blue')

plt.savefig(save_path)