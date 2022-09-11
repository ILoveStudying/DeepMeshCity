import matplotlib.pyplot as plt


def Plot_TokyoChannel(c16, c32, c64, c128, Y_S, Y_E, savefig):
    plt.figure(figsize=(7, 5))
    X = ['MSE', 'MAE×10']

    bar_width = 0.2
    bar1 = list(range(len(X)))
    bar2 = [i + bar_width for i in bar1]
    bar3 = [i + 2 * bar_width for i in bar1]
    bar4 = [i + 3 * bar_width for i in bar1]

    plt.tick_params(labelsize=13)
    plt.xticks(bar2, X, fontsize=14)

    # color='pink',color='teal'
    plt.ylim(Y_S, Y_E)
    plt.bar(X, c16, width=bar_width, label='Channel=16', color='paleturquoise')
    plt.bar(bar2, c32, width=bar_width, label='Channel=32', color='turquoise')
    plt.bar(bar3, c64, width=bar_width, label='Channel=64', color='darkturquoise')
    plt.bar(bar4, c128, width=bar_width, label='Channel=128', color='darkcyan')
    for i in range(len(X)):
        plt.text(bar1[i] - 0.10, c16[i] + 0.09, c16[i], weight="bold", fontsize=11)
        plt.text(bar2[i] - 0.09, c32[i] + 0.09, c32[i], weight="bold", fontsize=11)
        plt.text(bar3[i] - 0.09, c64[i] + 0.09, c64[i], weight="bold", fontsize=11)
        plt.text(bar4[i] - 0.10, c128[i] + 0.09, c128[i], weight="bold", fontsize=11)

    plt.legend(fontsize=13, loc='best')
    plt.savefig(savefig + '.png', dpi=300, bbox_inches='tight')
    plt.show()


def Plot_TokyoBlock(b1, b2, b3, b4, Y_S, Y_E, savefig):
    plt.figure(figsize=(7, 5))
    X = ['MSE', 'MAE×10']
    bar_width = 0.2
    bar1 = list(range(len(X)))
    bar2 = [i + bar_width for i in bar1]
    bar3 = [i + 2 * bar_width for i in bar1]
    bar4 = [i + 3 * bar_width for i in bar1]

    plt.tick_params(labelsize=13)
    plt.xticks(bar2, X, fontsize=14)

    # ,color='pink',color='teal'
    plt.ylim(Y_S, Y_E)
    plt.bar(X, b1, width=bar_width, label='Block=1', color='paleturquoise')
    plt.bar(bar2, b2, width=bar_width, label='Block=2', color='turquoise')
    plt.bar(bar3, b3, width=bar_width, label='Block=3', color='darkturquoise')
    plt.bar(bar4, b4, width=bar_width, label='Block=4', color='darkcyan')
    for i in range(len(X)):
        plt.text(bar1[i] - 0.11, b1[i] + 0.09, b1[i], weight="bold", fontsize=11)
        plt.text(bar2[i] - 0.09, b2[i] + 0.09, b2[i], weight="bold", fontsize=11)
        plt.text(bar3[i] - 0.12, b3[i] + 0.09, b3[i], weight="bold", fontsize=11)
        plt.text(bar4[i] - 0.10, b4[i] + 0.09, b4[i], weight="bold", fontsize=11)

    plt.legend(fontsize=13, loc='upper left')
    plt.savefig(savefig + '.png', dpi=300, bbox_inches='tight')
    plt.show()


def TokyoChannel():
    Tokyochannel16 = [32.446, 33.96]
    Tokyochannel32 = [30.607, 32.84]
    Tokyochannel64 = [29.670, 32.28]
    Tokyochannel128 = [31.172, 33.03]
    Plot_TokyoChannel(Tokyochannel16, Tokyochannel32, Tokyochannel64, Tokyochannel128, 28, 36, 'Tokyochannel')


def TokyoBlock():
    Tokyoblock1 = [31.046, 33.14]
    Tokyoblock2 = [29.670, 32.28]
    Tokyoblock3 = [30.891, 32.76]
    Tokyoblock4 = [30.651, 32.77]
    Plot_TokyoBlock(Tokyoblock1, Tokyoblock2, Tokyoblock3, Tokyoblock4, 28, 35, 'Tokyoblock')


TokyoChannel()
TokyoBlock()
