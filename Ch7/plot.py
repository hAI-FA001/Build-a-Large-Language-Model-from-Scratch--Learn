import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    f, ax1 = plt.subplots(figsize=(5,3))

    ax1.plot(epochs_seen, train_losses, label='Training Loss')
    ax1.plot(epochs_seen, val_losses, label='Val Loss', linestyle='-.')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only integer labels

    ax2 = ax1.twiny()  # share the same Y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # invisible plot to align ticks
    ax2.set_xlabel('Tokens Seen')

    f.tight_layout()
    plt.savefig('loss-plot.pdf')
    plt.show()