

class FinancialModel:
    def __init__(self, repayment_rate, value_of_loan, cogs=-92, mean_acreage=1.35, farmer_price_per_acre=124,
                 acquisition_cost=-7.5):
        """
        Positive values always taken to be those that repaid
        :param repayment_rate:
        :param value_of_loan:
        :param cogs:
        :param mean_acreage:
        :param farmer_price_per_acre:
        :param acquisition_cost:
        """
        self.repayment_rate = repayment_rate
        self.cogs = cogs
        self.mean_acreage = mean_acreage
        self.farmer_price_per_acre = farmer_price_per_acre
        self.value_of_loan = value_of_loan
        self.acquisition_cost = acquisition_cost

    @property
    def rp(self):
        return self.repayment_rate

    @property
    def rn(self):
        return 1 - self.repayment_rate

    @property
    def utp(self):
        return self.mean_acreage * (self.cogs + self.farmer_price_per_acre) + self.acquisition_cost + self.value_of_loan

    @property
    def ufp(self):
        return self.mean_acreage * self.cogs + self.acquisition_cost + self.value_of_loan

    @property
    def utn(self):
        return self.acquisition_cost

    @property
    def ufn(self):
        return self.acquisition_cost

    @property
    def slope(self):
        return (self.utn - self.ufp) / (self.utp - self.ufn) * (self.rn / self.rp)

    def utility(self, TPR, FPR):
        """
        Utility as a function of true positive rate and false positive rate
        :param TPR: True positive rate
        :param FPR: False positive rate
        :return:
        """
        return (self.rp * self.utp * TPR) + (self.rp * self.ufn * (1-TPR)) + (
                self.rn * self.utn * (1-FPR)) + (self.rn * self.ufp * FPR)

    def zero_utility_indifference(self, FPR):
        """
        Calculate the TPR required for utility indifference for a given FPR
        :param FPR:
        :return TPR:
        """
        return self.slope*FPR - (self.ufn * self.rp + self.utn * self.rn) / ( (self.utp - self.ufn) * self.rp)

    def utility_from_roc_pt(self, FPR, TPR):
        """
        Returns the utility value for any point in (FPR,TPR) space
        :param FPR:
        :param TPR:
        :return:
        """
        intercept = TPR - self.slope*FPR
        return self.utility_from_intercept(intercept)

    def utility_from_intercept(self, intercept):
        """
        Utility of intercept (e.g. TPR with FPR=0)
        :param intercept:
        :return:
        """
        return self.rp * (self.utp - self.ufn) * intercept + self.ufn * self.rp + self.utn * self.rn

    def intercept_from_utility(self, utility):
        return (utility - ( self.ufn * self.rp + self.utn * self.rn) ) / ( self.rp * (self.utp - self.ufn) )

    def max_utility_from_roc(self, FPR, TPR):
        """
        Calculates the max utility from an ROC curve
        :param iterable FPR: false positive rate
        :param iterable TPR: true positive rate
        :return:
        """
        return max(map(lambda x: self.utility_from_roc_pt(x[0], x[1]), zip(FPR, TPR)))

    def utility_from_confusion_matrix(self, confusion_mat):
        fpr = confusion_mat[0,1] / confusion_mat[0,:].sum()
        tpr = confusion_mat[1,1] / confusion_mat[1,:].sum()
        return self.utility_from_roc_pt(fpr, tpr)


