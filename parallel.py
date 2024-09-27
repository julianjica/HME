import numpy as np
from subprocess import run
import sys

if __name__ == "__main__":
    sigmad = np.linspace(1, 10, 100)
    run(["./bobyqa.out", str(3), str(sigmad[int(sys.argv[1])])])

