from dataclasses import dataclass

import matplotlib.pylab as plt
from sklearn.metrics import roc_curve
import numpy as np


@dataclass
class UtilityMatrix:
    """Utility for each square in a confusion matrix"""
    u_tp: float
    u_fp: float
    u_tn: float
    u_fn: float
    prevalence: float # must be between 0 and 1

    @property
    def positivity(self):
        return self.prevalence / (1 - self.prevalence)

    @property
    def negativity(self):
        return 1 / self.positivity

    @property
    def slope(self):
        numerator = (self.u_tn - self.u_fn) * self.negativity
        denominator = (self.u_tp - self.u_fn) * self.positivity
        return numerator / denominator

    def intercept(self, utility: float):
        # TODO double check this math
        numerator = utility - self.u_fn * self.positivity - self.u_tn * self.negativity
        denominator = (self.u_tp - self.u_fn) * self.positivity
        return numerator / denominator

    def tpr_from_fpr_and_prevalence(self, fpr: float):
        pass


class ROCPlotter:

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.05])
        self.ax.set_xlabel('False Positive Rate')
        self.ax.set_ylabel('True Positive Rate')

    def plot_precision(self, precision: float, prevalence: float = 0.5):
        """Add line of a given precision.
        TPR = FPR / (r - precision)
        where r is the positivity ratio (P/N)
        Prevalence = P / (P + N)"""
        r = prevalence / (1 - prevalence)
        fpr_values = np.linspace(0, 1, 5)
        tpr_values = fpr_values /  (r * (1 - precision))
        self.ax.plot(fpr_values, tpr_values, label=f'Precision: {precision:0.2f}')

    def plot_iso_utility(self, utility_matrix: UtilityMatrix, prevalence: float =  0.5):
        """Add line of iso-utility"""
        r = prevalence / (1 - prevalence)
        fpr_values = np.linspace(0,1,5)


    def show(self):
        self.ax.legend()
        self.fig.tight_layout()
        return self.fig, self.ax



