import pickle
import numpy as np
import matplotlib.pyplot as plt

t_10 = "target_reward_1"
t_100 = "target_reward_1000"
t_1000 = "target_reward_10000"
t_10000 = "target_reward_100000"

epi = "epi_1"

with open(t_10, "rb") as f:
	tar_10 = pickle.load(f)
with open(t_100, "rb") as f:
	tar_100 = pickle.load(f)
with open(t_1000, "rb") as f:
	tar_1000 = pickle.load(f)
with open(t_10000, "rb") as f:
	tar_10000 = pickle.load(f)
with open(epi, "rb") as f:
	ep = pickle.load(f)

tar_10 = np.array(tar_10)[1:]
tar_100 = np.array(tar_100)[1:]
tar_1000 = np.array(tar_1000)[1:]
tar_10000 = np.array(tar_10000)[1:]
ep = np.array(ep)[1:]

plt.plot(ep, tar_10, label="target 1")
plt.plot(ep, tar_100, label="target 1000")
plt.plot(ep, tar_1000, label="target 10000")
plt.plot(ep, tar_10000, label="target 100000")
plt.legend(loc='lower right')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward under different target update strategy [Alien]")
plt.tight_layout()
plt.savefig("ddqnvs.png")
