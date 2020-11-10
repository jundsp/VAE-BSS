import torch
import numpy as np
import matplotlib.pyplot as plt

x = torch.randn(100)
plt.plot(x)
plt.savefig('results/img.png')
plt.show()