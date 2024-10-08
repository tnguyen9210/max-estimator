
import unittest

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import numpy as np
import scipy.stats as stats

def main():
    suite = unittest.TestSuite()
    # suite.addTest(TestNormalDist1d("test_visualizing_pdf_and_cdf"))
    suite.addTest(TestNormalDist1d("test_estimating_pdf"))
    # suite.addTest(TestNormalDist1d("test_sns_kdeplot"))
    
    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)

    
class TestNormalDist1d(unittest.TestCase):
    """
    """
    def setUp(self):
        np.random.seed(0)
        
        # set sns color palette
        sns.set()
        sns.set_palette("tab10")

    def test_visualizing_pdf_and_cdf(self):
        """ Visualize pdf and cdf of various normal dist (of different params)
        generated by stats.norm(mu, sigma).pdf and stats.norm(mu, sigma).cdf
        """
        # params
        num_bins = 100
        normal_dist_params = [[0.0, 1.0], [0.0, 2.0], [0.0, 0.5], [1.0, 1.0]]

        # plot
        fig, axes = plt.subplots(2, 1, sharey='row')
        axes = axes.flatten()
        for i, (mu, sigma) in enumerate(normal_dist_params):
            # set normal dist
            normal_dist = stats.norm(mu, sigma)

            # compute pdf and cdf
            x = np.linspace(mu-3*sigma, mu+3*sigma, num_bins)
            px = normal_dist.pdf(x)
            cx = normal_dist.cdf(x)
        
            axes[0].plot(x, px, label=f"N( {mu:1.1f}, {sigma:1.1f} )")
            axes[1].plot(x, cx, label=f"N( {mu:1.1f}, {sigma:1.1f} )")
            axes[0].legend()
            axes[1].legend()
            
        plt.show()

    def load_data(self, mu=0.0, sigma=1.0, num_bins=50, num_data=10000):
        # data_x = mu + sigma*np.random.randn(num_data,)
        data_x = np.random.normal(mu, sigma, size=(num_data,))
        x_bins = np.linspace(mu-3*sigma, mu+3*sigma, num_bins)
        return data_x, x_bins

    def test_estimating_pdf(self):
        """ 2 Ways of estimating density of 1d normal dist: 
        computing histogram and computing kde 
        """
        # load data
        mu = 0.0
        sigma = 1.0
        data_x, x_bins = self.load_data(mu, sigma)
        x = (x_bins[:-1] + x_bins[1:]) / 2  # centerize

        # true dist
        px_true = stats.norm(mu, sigma).pdf(x)

        # 1st method: computing histogram and plt #
        # compute histogram, the number of examples in every bin
        hist, x_bins = np.histogram(data_x, bins=x_bins, density=False)

        # 2nd method: computing kernel density estimate function #
        kde_fn = stats.gaussian_kde(data_x)
        px_kde = kde_fn(x)

        # plt.bar(x, height=hist, width=(x_bins[1]-x_bins[0]))
        plt.plot(x, px_true, label="true density")
        plt.plot(x, hist, label="comp. histogram")
        plt.plot(x, px_kde, label="comp. kde")
        plt.legend()
        plt.show()
        
    def test_sns_kdeplot(self):
        """ Plot histogram of 1d normal dist directly using sns.kdeplot
        """
        data_x, _ = self.load_data()
        sns.kdeplot(data_x, fill=False)
        plt.show()

    
        
    


if __name__ == '__main__':
    main()


