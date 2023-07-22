import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.io import mmwrite
import os
import PetscBinaryIO


def grid_plot(u):
    fig = plt.figure()
    nb_rows = u.shape[1]
    gs = GridSpec(nb_rows, 1)
    for i in range(nb_rows):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(range(u.shape[0]), u[:, i], color="blue")
        ax.set_ylabel(f"weight dim : {i}")
    ax.set_xlabel("t")
    return fig


def plt_line(u):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(u)


def read_mat(file):
    __import__('pdb').set_trace()
    io = PetscBinaryIO.PetscBinaryIO()
    fh = open(file)
    objecttype = io.readObjectType(fh)
    if objecttype == 'Mat':
        v = io.readMatSciPy(fh)
        mmwrite('test.mtx', v, precision=30)
        os.remove(file)
        os.remove(file+'.info')


if __name__ == "__main__":
    actions = np.loadtxt('./ns_cylinder/actions.txt')
    observations = np.loadtxt('./ns_cylinder/observations.txt')
    energies = np.loadtxt('./ns_cylinder/energies.txt')
    fig = grid_plot(actions)
    fig = grid_plot(observations)
    plt_line(energies)
    read_mat('./ns_cylinder/amat/A.dat')
    plt.show()
