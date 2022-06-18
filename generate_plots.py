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


def make_plot_1():
    utility_matrix = UtilityMatrix(110, -162, -10, -10, 0.5)
    plotter = ROCPlotter()
    plotter.plot_iso_utility(utility_matrix, utility_matrix.utility_at_accept,
                             color=my_mapper(utility_matrix.utility_at_accept)). \
        plot_iso_utility(utility_matrix, utility_matrix.utility_at_reject,
                         color=my_mapper(utility_matrix.utility_at_reject)). \
        plot_iso_utility(utility_matrix, 5, color=my_mapper(10)). \
        plot_iso_utility(utility_matrix, 20, color=my_mapper(20))
    plotter.ax.set_title(f'Slope: {utility_matrix.slope:0.2f} ')
    plotter.show()
    # ax.title = f'Slope: {utility_matrix.slope}'
    plotter.fig.savefig('./iso_utility_1.png')
    plotter.fig.show()


def make_plot_2():
    utility_matrix = UtilityMatrix(110, -162, -10, -10, 0.6)
    plotter = ROCPlotter()
    plotter.plot_iso_utility(utility_matrix, utility_matrix.utility_at_accept,
                             color=my_mapper(utility_matrix.utility_at_accept)). \
        plot_iso_utility(utility_matrix, utility_matrix.utility_at_reject,
                         color=my_mapper(utility_matrix.utility_at_reject)). \
        plot_iso_utility(utility_matrix, 10, color=my_mapper(10)). \
        plot_iso_utility(utility_matrix, 20, color=my_mapper(20))
    plotter.ax.set_title(f'Slope: {utility_matrix.slope:0.2f} ')
    plotter.show()
    # ax.title = f'Slope: {utility_matrix.slope}'
    plotter.fig.savefig('./iso_utility_2.png')
    plotter.fig.show()


def make_single_lines_1():
    utility_matrix = UtilityMatrix(110, -162, -10, -10, 0.5)
    plotter = ROCPlotter()
    plotter. \
        plot_iso_utility(utility_matrix, utility_matrix.utility_at_reject,
                         color=my_mapper(utility_matrix.utility_at_reject))
    plotter.ax.set_title(f'Slope: {utility_matrix.slope:0.2f} ')
    plotter.show()
    # ax.title = f'Slope: {utility_matrix.slope}'
    plotter.fig.savefig('./single_1.png')
    plotter.fig.show()

def make_plots(utility_matrix, utility_values):
    plotter = ROCPlotter()
    for value in utility_values:
        plotter.plot_iso_utility(utility_matrix, value, color=my_mapper(value))
    plotter.ax.set_title(f'Slope: {utility_matrix.slope:0.2f} ')
    plotter.show()
    return plotter


if __name__ == '__main__':
    make_blank_plot()
    make_plot_1()
    make_plot_2()
    make_single_lines_1()
