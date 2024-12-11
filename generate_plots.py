from utils.roc_plotter import ROCPlotter, UtilityMatrix
import matplotlib.pylab as plt
import matplotlib as mpl

viridis = mpl.cm.get_cmap('viridis')


def color_mapper(min_val, max_val, scale=1):
    def c(num):
        val = (num - min_val) / (max_val - min_val) * scale
        return viridis(val)

    return c


my_mapper = color_mapper(-28, 25)


def make_blank_plot():
    plotter = ROCPlotter()
    plotter.show()
    plotter.fig.savefig('./blank.png')


def make_plots(utility_matrix, utility_values):
    plotter = ROCPlotter()
    for value in utility_values:
        plotter.plot_iso_utility(utility_matrix, value, color=my_mapper(value))
    plotter.ax.set_title(f'Slope: {utility_matrix.slope:0.2f} ')
    plotter.show()
    return plotter


def make_and_save_plot_0():
    utility_matrix = UtilityMatrix(110, -162, -10, -10, 0.5)
    plotter = make_plots(utility_matrix,
                         [utility_matrix.utility_at_accept,
                          utility_matrix.utility_at_reject,
                          0, 5, 10, 15, 20, 25])
    plotter.fig.savefig('./iso_utility_0.png')
    plotter.fig.show()


def make_and_save_plot_1():
    utility_matrix = UtilityMatrix(110, -162, -10, -10, 0.5)
    plotter = make_plots(utility_matrix,
                         [utility_matrix.utility_at_accept,
                          utility_matrix.utility_at_reject,
                          5, 20])
    plotter.fig.savefig('./iso_utility_1.png')
    plotter.fig.show()


def make_and_save_plot_2():
    utility_matrix = UtilityMatrix(110, -162, -10, -10, 0.6)
    plotter = make_plots(utility_matrix,
                         [utility_matrix.utility_at_accept,
                          utility_matrix.utility_at_reject,
                          10, 20])
    plotter.fig.savefig('./iso_utility_2.png')
    plotter.fig.show()


def make_and_save_plot_3():
    utility_matrix = UtilityMatrix(110, -162, -10, -10, 0.5)
    plotter = make_plots(utility_matrix,
                         [utility_matrix.utility_at_accept,
                          utility_matrix.utility_at_reject,
                          1, 10, 20])
    plotter.fig.savefig('./iso_utility_3.png')
    plotter.fig.show()

make_and_save_plot_3()


if __name__ == '__main__':
    make_blank_plot()
    make_and_save_plot_0()
    make_and_save_plot_1()
    make_and_save_plot_2()
    make_and_save_plot_3()
    utility_matrix = UtilityMatrix(110, -162, -10, -10, 0.5)
    p = make_plots(utility_matrix, [utility_matrix.utility_at_reject])
    p.fig.savefig('./basic_1.png')
    p.fig.show()
    p = make_plots(utility_matrix, [utility_matrix.utility_at_reject,
                                    utility_matrix.utility_at_accept])
    p.fig.savefig('./basic_2.png')
    p.fig.show()
