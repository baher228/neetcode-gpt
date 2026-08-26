import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
      z-=np.max(z)
      ze = np.exp(z)
      return np.round(ze / np.sum(ze), 4)
