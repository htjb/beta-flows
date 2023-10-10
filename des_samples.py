import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from anesthetic import read_chains
from anesthetic import make_2d_axes


samples = read_chains('DES_samples/DES')
samples = samples.compress()
names =[samples.columns[:6].values[i][0] for i in range(len(samples.columns[:6].values))]

"""betas = [0.001, 0.1, 0.25, 0.75, 1.]
for i in range(len(betas)):
    print(betas[i])
    s = samples.set_beta(betas[i])
    s.plot_2d(names)
    plt.show()"""

from beta_flow.flow import MAF

f = MAF(samples)
f.train(10000, early_stop=True)

plt.plot(f.loss_history)
plt.plot(f.test_loss_history)
plt.show()