import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io


def construct(k, A):
    showing = np.zeros(A.shape)
    u, s, v = np.linalg.svd(A)
    for i in range(k):
        showing += s[i] * (np.outer(u[:, i], v[i, :]))
    return showing


fname = "test.jpg"
img = io.imread(fname, as_gray=True)
# plt.imshow(img, cmap="gray")
# plt.show()

k = 25
rows = int(k / 5)
columns = 5
step = 2

plt.rcParams['figure.figsize'] = (10.0, 20.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
for i in range(k):
    plt.subplot(rows, columns, i + 1)
    plt.title(f"rank = {step * i + 1}")
    plt.axis('off')
    plt.imshow(construct(step * i + 1, img), cmap="gray")

plt.show()
