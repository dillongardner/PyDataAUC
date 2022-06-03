import matplotlib.pylab as plt
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from pandas.core.generic import NDFrame
import numpy as np
import functools


class ROC:
    """
    ROC curve class
        :param array-like y_pred:
        :param array-like y_true:
        :param int num_samples: number of bootstrap samples
    """

    def __init__(self, y_pred, y_true, num_samples=100, is_quantized=False):
        self.y_pred = y_pred.values if isinstance(y_pred, NDFrame) else y_pred
        self.y_true = y_true.values if isinstance(y_true, NDFrame) else y_true
        if len(y_pred) != len(y_true):
            raise ValueError('y_pred and y_true are different lengths ({} and {})'.format(len(y_pred), len(y_true)))
        self.num_samples = num_samples
        self._raw_roc = None
        self._size = len(y_pred)
        self.is_quantized = is_quantized
        self.interp_method = 'nearest' if self.is_quantized else 'linear'

    @staticmethod
    def from_model(model, x, y, num_samples=100, is_quantized=False):
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(x)[:, 1]
        else:
            y_pred = model.predict(x)
        return ROC(y_pred, y, num_samples, is_quantized)

    @functools.lru_cache(maxsize=1)
    def roc_samples(self):
        def single_sample():
            samples = np.random.randint(low=0, high=self._size, size=self._size)
            roc = roc_curve(self.y_true[samples], self.y_pred[samples])
            return roc

        return np.array([single_sample() for _ in range(self.num_samples)])

    @property
    def _positive_rate(self):
        return self.y_true.sum() / len(self.y_true)

    @property
    def _negative_rate(self):
        return 1 - self._positive_rate

    def acceptance_rate(self, threshold):
        positives = self.tpr_from_thresh(threshold) * self._positive_rate
        negatives = self.fpr_from_thresh(threshold) * self._negative_rate
        return positives + negatives

    def positive_selected_rate(self, threshold):
        positives = self.tpr_from_thresh(threshold) * self._positive_rate
        negatives = self.fpr_from_thresh(threshold) * self._negative_rate
        return positives / (positives + negatives)

    @property
    def tpr_from_fpr(self):
        return lambda fpr: np.array([interp1d(roc[0], roc[1], kind=self.interp_method)(fpr)
                                     for roc in self.roc_samples()])

    @property
    def tpr_from_thresh(self):
        return lambda threshold: np.array(
            [interp1d(roc[2], roc[1], bounds_error=False, fill_value=(1, 0), kind=self.interp_method)(threshold)
             for roc in self.roc_samples()])

    @property
    def fpr_from_thresh(self):
        return lambda threshold: np.array(
            [interp1d(roc[2], roc[0], bounds_error=False, fill_value=(1, 0), kind=self.interp_method)(threshold)
             for roc in self.roc_samples()])

    @property
    def thresh_from_fpr(self):
        return lambda fpr: np.array(
            [interp1d(roc[0], roc[2], bounds_error=False, fill_value=(1, 0), kind=self.interp_method)(fpr)
             for roc in self.roc_samples()])

    @property
    def thresh_from_tpr(self):
        return lambda tpr: np.array(
            [interp1d(roc[1], roc[2], bounds_error=False, fill_value=(1, 0), kind=self.interp_method)(tpr)
             for roc in self.roc_samples()])

    @property
    def _min_threshold(self):
        return min([min(r[2]) for r in self.roc_samples()])

    @property
    def _max_threshold(self):
        return max([max(r[2]) for r in self.roc_samples()])

    def utility_from_threshold(self, financial_model, threshold):
        """
        Returns function that takes a threshold and outputs an array of estimates of utility
        :param FinancialModel financial_model:
        :param threshold
        :return:
        """
        tpr = self.tpr_from_thresh(threshold)
        fpr = self.fpr_from_thresh(threshold)
        u = financial_model.utility(tpr, fpr)
        return u


    @property
    def raw_roc(self):
        if self._raw_roc is None:
            self._raw_roc = roc_curve(self.y_true, self.y_pred)
        return self._raw_roc

    def make_bootstrapped_roc(self):
        samples = np.random.randint(low=0, high=self._size, size=self._size)
        roc = roc_curve(self.y_true[samples], self.y_pred[samples])
        return interp1d(roc[0], roc[1])

    def get_bootstrap_rocs(self, num_samples=100):
        return np.array([self.make_bootstrapped_roc() for _ in range(num_samples)])

    def _percentile_roc(self, x, q):
        return np.percentile(self.tpr_from_fpr(x), q)

    @property
    def percentile_roc(self):
        return np.vectorize(self._percentile_roc, otypes=[np.float])

    @property
    def mean_roc(self):
        return np.vectorize(lambda x: np.mean(self.tpr_from_fpr(x)), otypes=[np.float])

    @property
    def roc_areas(self):
        x_val = np.linspace(0, 1, 200)
        y_values = np.array([self.tpr_from_fpr(x) for x in x_val]).reshape(self.num_samples, len(x_val))
        return y_values.sum(axis=1) / len(x_val)

    def plot(self, lower_thresh=16, upper_thresh=84, steps=50, title='ROC', ax=None):
        """
        Plot ROC
        :param float lower_thresh: lower confidence bound (percent)
        :param float upper_thresh: upper confidence bound (percent)
        :param int steps: steps used in interpolation
        :param str title: title of plot
        :param Axes ax: If none, will create new plot"""
        x_val = np.linspace(0, 1, steps)
        mean_val = self.mean_roc(x_val)
        upper_val = self.percentile_roc(x_val, upper_thresh)
        lower_val = self.percentile_roc(x_val, lower_thresh)
        mean_area = np.sum(mean_val) / steps
        if ax is None:
            ax = plt.gca()
        fig = ax.figure
        lw = 2
        ax.plot(x_val, mean_val, color='darkorange',
                lw=lw, label='Mean ROC: AUC {0:.3f}'.format(mean_area))
        ax.plot(x_val, upper_val, color='black', lw=lw / 2, linestyle='--',
                label='{0} Percent Range'.format(upper_thresh - lower_thresh))
        ax.plot(x_val, lower_val, color='black', lw=lw / 2, linestyle='--')
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        fig.tight_layout()
        return fig, ax

    def max_utility(self, financial_model, steps=50):
        """
        Calculate the maximum utility based on financial model
        :param FinancialModel financial_model: model with financial assumptions
        :param int steps: number of steps to try
        :return:
        """
        fpr = np.linspace(0, 1, steps)
        mean_tpr = self.mean_roc(fpr)
        return financial_model.max_utility_from_roc(fpr, mean_tpr)

    def plot_with_max_utility(self, financial_model, lower_thresh=16, upper_thresh=84, steps=50,
                              title='ROC with Utility', ax=None):
        """
        Plot ROC with utility
        :param FinancialModel financial_model:
        :param float lower_thresh:
        :param float upper_thresh:
        :param int steps:
        :param str title: Title for plot
        :param Axes ax:
        :return:
        """
        fig, ax = self.plot(lower_thresh=lower_thresh, upper_thresh=upper_thresh, steps=steps, title=title, ax=ax)
        max_utility = self.max_utility(financial_model)
        x_values = np.linspace(0, 1, 5)
        y_values = financial_model.slope * x_values + financial_model.intercept_from_utility(max_utility)
        ax.plot(x_values, y_values, label='Max Utility: {0:.2f}'.format(max_utility))
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_threshold(self, financial_model, lower_thresh=16, upper_thresh=84, steps=100, min_val=None, max_val=None):
        min_thresh = min_val if min_val is not None else self._min_threshold
        max_thresh = max_val if max_val is not None else self._max_threshold
        x_val = np.linspace(min_thresh, max_thresh, steps)
        y_approval = self.acceptance_rate(x_val)
        y_tpr = self.tpr_from_thresh(x_val)
        y_fpr = self.tpr_from_thresh(x_val)
        y_utility = self.utility_from_threshold(financial_model, x_val)
        fig, axes = plt.subplots(4, sharex=True, figsize=(5, 8))
        def plot_axes(ax, y, y_label):
            lw=2
            ax.plot(x_val, y.mean(axis=0), color='darkorange',
                    lw=lw, label='Mean')
            ax.plot(x_val, np.percentile(y, upper_thresh, axis=0), color='black', lw=lw / 2, linestyle='--',
                    label='{0} Percent Range'.format(upper_thresh - lower_thresh))
            ax.plot(x_val, np.percentile(y, lower_thresh, axis=0), color='black', lw=lw / 2, linestyle='--')
            ax.set_ylabel(y_label)
            return ax
        approval_ax = plot_axes(axes[0], y_approval, 'Approval Rate')
        approval_ax.legend()
        utility_ax = plot_axes(axes[1], y_utility, 'Utility')
        tpr_ax =plot_axes(axes[2], y_tpr, 'True Positive Rate')
        fpr_ax =plot_axes(axes[3], y_fpr, 'False Positive Rate')
        fpr_ax.set_xlabel('Threshold')
        return fig, axes


