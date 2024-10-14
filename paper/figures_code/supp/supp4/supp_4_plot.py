import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(sys.path[0], "../../../code/bandit")))
import matplotlib.pyplot as plt
import seaborn as sns

# --- Specify parameters ---

# save path
load_path = os.path.abspath(os.path.join(sys.path[0], "../../../figures/supp/supp4/"))


def plot_violin(data, position, edgecolour, facecolour):

    vparts = plt.violinplot(dataset=data, positions=[position], showmeans=True)
    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        vp = vparts[partname]
        vp.set_edgecolor(edgecolour)
        vp.set_linewidth(1)

    for vp in vparts["bodies"]:
        vp.set_facecolor(facecolour)
        vp.set_edgecolor(facecolour)
        vp.set_linewidth(1)
        vp.set_alpha(0.5)

    return None


# --- Main function for replay ---
def main():

    prop_matrix = np.load(os.path.join(load_path, "prop_matrix.npy"))
    num_matrix = np.load(os.path.join(load_path, "num_matrix.npy"))

    plt.figure(figsize=(8, 4), dpi=100, constrained_layout=True)
    plt.subplot(121)
    plot_violin(prop_matrix[:, 0], 1, "black", "black")
    plt.errorbar(
        [1],
        np.mean(prop_matrix[:, 0]),
        np.sqrt(np.var(prop_matrix[:, 0]) / len(prop_matrix[:, 0])),
        c="orange",
    )
    plt.ylim(0.0, 1.0)
    plt.xlim(0.5, 1.5)
    plt.xticks([])
    plt.ylabel("Proportion of forward to reverse sequences", fontsize=12)

    plt.subplot(122)
    plot_violin(num_matrix[:, 0], 1, "black", "purple")
    plt.errorbar(
        [1],
        np.mean(num_matrix[:, 0]),
        np.sqrt(np.var(num_matrix[:, 0]) / len(num_matrix[:, 0])),
        c="orange",
    )
    plot_violin(num_matrix[:, 1], 2, "black", "green")
    plt.errorbar(
        [2],
        np.mean(num_matrix[:, 1]),
        np.sqrt(np.var(num_matrix[:, 1]) / len(num_matrix[:, 1])),
        c="orange",
    )
    plt.xticks([1, 2], ["seqs", "noseqs"], fontsize=12)
    plt.xlim(0, 3)
    plt.ylim(0, np.max(num_matrix[:] + 25))
    plt.ylabel("Number of replayed actions", fontsize=12)

    plt.savefig(os.path.join(load_path, "supp_4.png"))
    plt.savefig(os.path.join(load_path, "supp_4.svg"), transparent=True)
    plt.close()

    return None


if __name__ == "__main__":
    main()
