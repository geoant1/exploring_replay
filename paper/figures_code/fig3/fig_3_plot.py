import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle, shutil
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../../code/maze')))
from utils import plot_maze, plot_need

data_path   = os.path.abspath(os.path.join(sys.path[0], '../../figures/fig3'))

with open(os.path.join(data_path, 'data', '0', 'ag.pkl'), "rb") as f:
    agent = pickle.load(f)

G = np.load(os.path.join(data_path, 'gain.npy'))
N = np.load(os.path.join(data_path, 'need.npy'))
S = np.load(os.path.join(data_path, 'states.npy'))

num_moves = 2000

def main():

    fig = plt.figure(figsize=(14, 8), constrained_layout=True, dpi=100)

    agent.barriers = [1, 1, 0]
    ax1 = fig.add_subplot(2, 3, 1)
    plot_need(ax1, S[0], agent, colormap='Blues', colorbar=True)
    ax1.set_title('State occupancy', fontsize=16)
    ax1.set_ylabel('Moves 1-%u'%num_moves, fontsize=16)
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax2 = fig.add_subplot(2, 3, 2)
    G_plot = G[0]
    G_plot[G_plot <= 0.] = np.nan
    G_plot /= np.nanmax(G_plot)
    
    plot_maze(ax2, G_plot, agent, colormap='Greens', colorbar=True)
    ax2.set_title('Normalised average max Gain', fontsize=16)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax3 = fig.add_subplot(2, 3, 3)
    plot_need(ax3, N[0], agent, colormap='Oranges', colorbar=True)
    ax3.set_title('Normalised average max Need', fontsize=16)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    agent.barriers = [0, 1, 0]
    ax4 = fig.add_subplot(2, 3, 4)
    plot_need(ax4, S[1], agent, colormap='Blues')
    ax4.set_ylabel('Moves %u-%u'%(num_moves, num_moves*2), fontsize=16)
    ax4.text(-0.1, 1.1, 'D', transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax5 = fig.add_subplot(2, 3, 5)
    G_plot = G[1]
    G_plot[G_plot <= 0.] = np.nan
    G_plot /= np.nanmax(G_plot)

    plot_maze(ax5, G_plot, agent, colormap='Greens')
    ax5.text(-0.1, 1.1, 'E', transform=ax5.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ax6 = fig.add_subplot(2, 3, 6)
    plot_need(ax6, N[1], agent, colormap='Oranges')
    ax6.text(-0.1, 1.1, 'F', transform=ax6.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    plt.savefig(os.path.join(data_path, 'fig_3.png'))
    plt.savefig(os.path.join(data_path, 'fig_3.svg'), transparent=True)
    plt.close()

    return None

if __name__ == '__main__':
    main()