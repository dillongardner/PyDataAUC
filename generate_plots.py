from utils.roc_plotter import ROCPlotter, UtilityMatrix
import matplotlib.pylab as plt
import numpy as np

utility_matrix = UtilityMatrix(100, -150, -10, -10, 0.5)

plotter = ROCPlotter()
plotter.show()
plotter.fig.savefig('./blank.png')

print(f'Slope of {utility_matrix} is {utility_matrix.slope:0.2f}')
plotter = ROCPlotter()
plotter.plot_iso_utility(utility_matrix, utility_matrix.utility_at_accept).\
    plot_iso_utility(utility_matrix, utility_matrix.utility_at_reject).\
    plot_iso_utility(utility_matrix, 5).\
    plot_iso_utility(utility_matrix, 20)
plotter.ax.set_title(f'Slope: {utility_matrix.slope:0.2f} ')
plotter.show()
#ax.title = f'Slope: {utility_matrix.slope}'
plotter.fig.savefig('./iso_utility_1.png')
plotter.fig.show()

utility_matrix.prevalence = 0.6
print(f'Slope of {utility_matrix} is {utility_matrix.slope:0.2f}')
plotter = ROCPlotter()
plotter.plot_iso_utility(utility_matrix, utility_matrix.utility_at_accept). \
    plot_iso_utility(utility_matrix, utility_matrix.utility_at_reject). \
    plot_iso_utility(utility_matrix, 10). \
    plot_iso_utility(utility_matrix, 20)
plotter.ax.set_title(f'Slope: {utility_matrix.slope:0.2f} ')
plotter.show()
#ax.title = f'Slope: {utility_matrix.slope}'
plotter.fig.savefig('./iso_utility_2.png')
plotter.fig.show()


if __name__ == '__main__':
    pass