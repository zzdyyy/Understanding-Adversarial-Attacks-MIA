import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import pandas as pd


MARKERS = ['x', 'o', '^', 'v', 'D', '<', '>']

titles = ['fgsm', 'bim', 'pgd']

# per attack method
sns.set_style("whitegrid")
fig = plt.figure(figsize=(20, 5))
gs = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.)
for i in range(3):
    ax = fig.add_subplot(gs[0, i])
    df = pd.read_csv('vis/acc_eps/%s.csv' % titles[i])
    ax.plot(df['epsilon'], 100*df['Fundoscopy'], marker=MARKERS[0], markersize=5, linewidth=1, label='Fundoscopy')
    ax.plot(df['epsilon'], 100*df['Chest X-Ray'], marker=MARKERS[1], markersize=5, linewidth=1, label='Chest X-Ray')
    ax.plot(df['epsilon'], 100*df['Dermoscopy'], marker=MARKERS[2], markersize=5, linewidth=1, label='Dermoscopy')
    ax.set_xlabel(r"Perturbation size epsilon ($\epsilon$)", fontsize=15)
    fig.gca().xaxis.set_ticks(np.arange(0, [11, 1.2, 1.2][i], [1, 0.2, 0.2][i]))
    fig.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(['%d/255', '%.1f/255', '%.1f/255'][i]))
    ax.set_xlim(0, [10, 1, 1][i])
    ax.set_ylabel("Accuracy (%)", fontsize=15)
    # ax.title("CIFAR-10 with clean labels", fontsize=16)
    ax.legend(loc='upper right', ncol=1, fontsize=13)  # lower/center right
    ax.set_ylim(bottom=0.0, top=100.0)
plt.savefig('vis/acc_eps/fig3.png', dpi=300)
plt.show()

# per dataset
titles = ['dr', 'cxr', 'derm']

cw = pd.read_csv('vis/acc_eps/cw-li.csv')
for i in range(3):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(5, 5))
    # gs = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.)
    ax = fig.gca() # ax = fig.add_subplot(gs[0, i])
    df = pd.read_csv('vis/acc_eps/%s.csv' % titles[i])
    ax.plot(df['epsilon'], 100*df['FGSM'], marker=MARKERS[0], markersize=5, linewidth=1, label='FGSM')
    ax.plot(list(0.1*df['epsilon']) + list(range(2,11)), list(100*df['BIM']) + [0.0]*9, marker=MARKERS[1], markersize=5, linewidth=1, label='BIM')
    ax.plot(list(0.1*df['epsilon']) + list(range(2,11)), list(100*df['PGD']) + [0.0]*9, marker=MARKERS[2], markersize=5, linewidth=1, label='PGD')
    ax.plot(cw['epsilon'], 100*cw[titles[i]], marker=MARKERS[3], markersize=5, linewidth=1, label='CW')
    # ax.plot(df['epsilon'], 100*df['FGSM_bb'], marker=MARKERS[0], markersize=5, linewidth=1, label='Blackbox FGSM')
    # ax.plot(list(0.1*df['epsilon']), list(100*df['BIM_bb']), marker=MARKERS[1], markersize=5, linewidth=1, label='BlackBox BIM')
    # ax.plot(list(0.1*df['epsilon']), list(100*df['PGD_bb']), marker=MARKERS[2], markersize=5, linewidth=1, label='BlackBox PGD')
    ax.set_xlabel(r"Perturbation size epsilon ($\epsilon$)", fontsize=20)
    fig.gca().xaxis.set_ticks(np.arange(0, 11, 1))
    fig.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d/255'))
    ax.set_xlim(0, 5)
    ax.set_ylabel("Accuracy (%)", fontsize=20)
    # ax.title("CIFAR-10 with clean labels", fontsize=16)
    ax.legend(loc='upper right', ncol=1, fontsize=20)  # lower/center right
    ax.set_ylim(bottom=-0.1, top=100.0)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.tight_layout()
    plt.savefig('vis/acc_eps/acceps_%s.png' % titles[i], dpi=300)
    plt.show()

def plot_test_acc_curve(models, model_names):
    sns.set_style("whitegrid")

    # plot initialization
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for i, model in enumerate(models):
        acc = np.load('log/acc_%s_cifar-10_0.npy' % model)
        acc = acc[1]
        xnew = np.arange(0, len(acc), 5)
        acc = acc[xnew]
        ax1.plot(xnew, acc, marker=MARKERS[i], markersize=5, linewidth=1, label='%s' % model_names[i])
        acc = np.load('log/acc_%s_cifar-10_40.npy' % model)
        acc = acc[1]
        acc = acc[xnew]
        ax2.plot(xnew, acc, marker=MARKERS[i], markersize=5, linewidth=1, label='%s' % model_names[i])

    ax1.set_xlabel("Epoch", fontsize=15)
    # ax1.xaxis.set_ticks(np.arange(0, 10, 1))
    ax1.set_ylabel("Test accuracy", fontsize=15)
    ax1.set_title("CIFAR-10 with clean labels", fontsize=16)
    ax1.legend(loc='lower right', ncol=2, fontsize=13)  # lower/center right
    ax1.set_ylim(bottom=0.0, top=1.0)

    ax2.set_xlabel("Epoch", fontsize=15)
    # ax2.xaxis.set_ticks(np.arange(0, 10, 1))
    ax2.set_ylabel("Test accuracy", fontsize=15)
    ax2.set_title("CIFAR-10 with 40% symmetric noisy labels", fontsize=16)
    ax2.legend(loc='lower right', ncol=2, fontsize=13)  # lower/center right
    ax2.set_ylim(bottom=0.0, top=1.0)


    fig.savefig("plots/test_acc_curve.png", dpi=300)
    plt.show()