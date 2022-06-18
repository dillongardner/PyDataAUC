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
    prevalence: float  # must be between 0 and 1

    @property
    def positivity(self) -> float:
        """Positivity is number of positive examples divided by the number of negative examples"""
        return self.prevalence / (1 - self.prevalence)

    @property
    def negativity(self) -> float:
        """Negativity is number of negative examples divided by the total number of examples"""
        return 1 / self.positivity

    @property
    def negavence(self) -> float:
        """Number of negative examples over total examples"""
        return 1 - self.prevalence

    @property
    def slope(self) -> float:
        numerator = (self.u_tn - self.u_fp) * self.negavence
        denominator = (self.u_tp - self.u_fn) * self.prevalence
        return numerator / denominator

    def intercept(self, utility: float) -> float:
        """
        Returns the Intercept for a given utility
        """
        numerator = utility - self.u_fn * self.prevalence - self.u_tn * self.negavence
        denominator = (self.u_tp - self.u_fn) * self.prevalence
        return numerator / denominator

    def tpr_from_fpr_utility(self, fpr: float, utility: float) -> float:
        """
        Returns the TPR for a given FPR and Utility
        """
        return self.slope * fpr + self.intercept(utility)

    @property
    def utility_at_reject(self) -> float:
        """Returns the utility value that occurs everything is reject
        This is (0,0) in ROC space"""
        return self.u_fn * self.prevalence + self.u_tn * self.negavence

    @property
    def utility_at_accept(self) -> float:
        """Returns the utility value that occurs everything is accepted
        This is (1,1) in ROC space"""
        return (1 - self.slope)*(self.u_tp - self.u_fn) * self.prevalence + self.utility_at_reject


class ROCPlotter:

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.05])
        self.ax.set_xlabel('False Positive Rate')
        self.ax.set_ylabel('True Positive Rate')

    def plot_precision(self, precision: float, prevalence: float = 0.5, **kwargs):
        """Add line of a given precision.
        TPR = FPR / (r - precision)
        where r is the positivity ratio (P/N)
        Prevalence = P / (P + N)"""
        r = prevalence / (1 - prevalence)
        fpr_values = np.linspace(0, 1, 5)
        tpr_values = fpr_values /  (r * (1 - precision))
        label = f'Precision: {precision:0.2f}\nPrevalence: {prevalence:0.2f}'
        self.ax.plot(fpr_values, tpr_values, label=label, **kwargs)
        return self

    def plot_iso_utility(self, utility_matrix: UtilityMatrix, utility: float, **kwargs):
        """Add line of iso-utility"""
        fpr_values = np.linspace(0,1,5)
        tpr_values =utility_matrix.tpr_from_fpr_utility(fpr_values, utility)
        self.ax.plot(fpr_values, tpr_values, label=f'Utility: {utility:0.0f}', **kwargs)
        return self

    def show(self):
        self.ax.legend()
        self.fig.tight_layout()
        return self



