import matplotlib.pylab as plt
from sklearn.metrics import roc_curve
import numpy as np


class ROCPlotter:

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.0])
        self.ax.set_xlabel('False Positive Rate')
        self.ax.set_ylabel('True Positive Rate')

    def plot_precision(self, precision: float, prevalence: float = 0.5):
        """Add line of a given precision.
        TPR = FPR / (r - precision)
        where r is the positivity ratio (P/N)"""
        r = prevalence / (1 - prevalence)
        fpr_values = np.linspace(0, 1, 5)
        tpr_values = fpr_values /  (r * (1 - precision))
        self.ax.plot(fpr_values, tpr_values, label=f'Precision: {precision:0.2f}')

    def show(self):
        self.ax.legend()
        self.fig.tight_layout()
        return self.fig, self.ax



