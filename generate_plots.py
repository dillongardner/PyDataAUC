from utils.roc_plotter import ROCPlotter
import matplotlib.pylab as plt

plotter = ROCPlotter()
plotter.plot_precision(0.5, 0.1)
plotter.plot_precision(0.9, 0.1)
plotter.show()

if __name__ == '__main__':
    plt.show()