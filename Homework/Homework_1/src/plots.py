import matplotlib.pyplot as plt

def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    #plt.show()
    plt.close()


def plot_torch(epochs,plottables,filename=None,ylim=None):
    plt.clf()
    plt.xlabel('Epoch')
    for label,plottable in plottables.items():
        plt.plot(epochs,plottable,label=label)
    plt.legend()
    if ylim :
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename)