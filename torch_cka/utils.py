from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt


def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def plot_results(results,
                save_path: str = None,
                title: str = None):

    hsic_matrix = results["CKA"]
    model1_name = results["model1_name"]
    model2_name = results["model2_name"]

    fig, ax = plt.subplots()
    im = ax.imshow(hsic_matrix, origin='lower', cmap='magma')
    ax.set_xlabel(f"Layers {model2_name}", fontsize=15)
    ax.set_ylabel(f"Layers {model1_name}", fontsize=15)

    if title is not None:
        ax.set_title(f"{title}", fontsize=18)
    else:
        ax.set_title(f"{model1_name} vs {model2_name}", fontsize=18)

    add_colorbar(im)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
