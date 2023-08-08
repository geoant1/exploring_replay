from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os, sys

params    = [[1, 1], [1, 2], [2, 1], [1, 3], [3, 1], [2, 2]]
x         = np.linspace(0, 1, 100)


save_path = os.path.abspath(os.path.join(sys.path[0], '../../figures/fig1/'))

def main():

    for idx, p in enumerate(params):

        a  = p[0]
        b  = p[1]

        rv = beta(a, b)

        plt.figure(figsize=(4, 3), dpi=100, constrained_layout=True)
        ax = plt.axes()
        ax.plot(x, rv.pdf(x), linewidth=10, c='black')
        ax.axvline(a/(a+b), linewidth=7, linestyle='dotted', c='r')
        ax.set_ylim(0-0.2, np.max(rv.pdf(x))+0.2)

        ax.tick_params(axis='both', which='both', colors='black', labelsize=45)
        ax.set_xlabel(r'$p$(reward)$=1$', fontsize=50)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        plt.savefig(os.path.join(save_path, 'prior_%u.svg'%idx), transparent=True)
        plt.savefig(os.path.join(save_path, 'prior_%u.png'%idx))
        plt.close()

if __name__ == '__main__':
    main()