import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def init_plotting(fig_size=(10, 7)):
    plt.style.reload_library()
    plt.style.use(["science", "grid"])
    sns.set_style("ticks")

    mpl.rcParams.update({
        'font.size': 14,
        'lines.linewidth': 2,
        'figure.figsize': (6, 6 / 1.61)
    })
    mpl.rcParams['grid.color'] = 'k'
    mpl.rcParams['grid.linestyle'] = ':'
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['lines.markersize'] = 6
    mpl.rcParams['lines.marker'] = None
    mpl.rcParams['axes.grid'] = True
    DEFAULT_FONTSIZE = 20
    mpl.rcParams.update({
        'font.size': DEFAULT_FONTSIZE,
        'lines.linewidth': 2,
        'legend.fontsize': DEFAULT_FONTSIZE,
        'axes.labelsize': DEFAULT_FONTSIZE,
        'xtick.labelsize': DEFAULT_FONTSIZE,
        'ytick.labelsize': DEFAULT_FONTSIZE,
        'figure.figsize': (7, 7.0 / 1.4)
    })

    markers_list = ["o", "x", "D", "*", "^"]
