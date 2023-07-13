import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from tester import Tester
import pandas as pd
import seaborn as sns
import torch

# plt.style.use('seaborn-white')
params = {"ytick.color": "black",
          "xtick.color": "black",
          "axes.labelcolor": "black",
          "axes.edgecolor": "black",
          "text.usetex": True,
          "font.family": "serif",
          "font.size": 40}
plt.rcParams.update(params)
legend_properties = {'weight': 'bold'}


class Visualizer(object):
    """Class for visualization functions."""

    def __init__(self,
                 tester: Tester,
                 ):
        self.model = tester.model
        self.data = tester.data.numpy()
        self.datadf = None
        self.tester = tester
        self.t = tester.t
        self.x = tester.x
        self.dt = tester.dt

    def dataframe_from_data(self):
        if self.datadf is None:
            m, n = self.data.shape
            out = np.empty((m, n, 3), dtype=self.data.dtype)
            out[..., 0] = self.t.unsqueeze(1)
            out[..., 1] = self.x.unsqueeze(0)
            out[..., 2] = self.data
            out.shape = (-1, 3)
            self.datadf = pd.DataFrame(out, columns=['t', 'd', 'value'])
        return self.datadf

    def bining(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(range(self.x.shape[0]), self.x)
        ax.set_xlabel('index')
        ax.set_ylabel('diameter')

    def surface(self, contour=True):
        t, x = np.meshgrid(self.t, self.x)
        fig = plt.figure()
        if contour:
            vmin = self.data.min()
            vmax = self.data.max()
            levels = np.around(np.linspace(vmin, vmax, 40), 2)
            ax = fig.add_subplot()
            contour = ax.contour(t,
                                 x,
                                 self.data.T,
                                 levels=levels,
                                 vmin=vmin,
                                 vmax=vmax)
            plt.colorbar(contour)
        else:
            ax = plt.axes(projection='3d')
            ax.plot_surface(t,
                            x,
                            self.data.T)
            ax.set_zlabel(r'$n(t,d)$')

        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$d$')

    def values_per_diameter(self, average=False, pow=False):
        if self.datadf is None:
            self.datadf = self.dataframe_from_data()
        plt.figure()
        df = self.datadf
        if average:
            df['value'] = df['value']
            df = self.datadf.groupby(['d']).agg('mean')
            ax = sns.scatterplot(data=df,
                                 x='d',
                                 y='value',
                                 marker='X',
                                 label='Average density')
            ax.set(yscale='log',
                   xscale='log',
                   title='Average value per diameter')
        else:
            ax = sns.scatterplot(data=df, x='d', y='value')
            ax.set(yscale='log',
                   xscale='log',
                   title='Values per diameter')
        if pow:
            y = self.tester.powerfunc(self.x, alpha=1.0, k=2/3)
            ax.plot(self.x, y, label=r'$d^{-2/3}$')
            x_ind = np.argwhere(self.x > 0.5)
            x_ = self.x[x_ind]
            y = self.tester.powerfunc(x_,
                                      alpha=1.0,
                                      k=10/3)
            ax.plot(x_.view(-1), y.view(-1), label=r'$d^{-10/3}$')
        ax.legend()

    def correlation_from_model(self,
                               inverted_trans=False,
                               corr=False,
                               transition=True,
                               show_colsum=True,
                               exp=True,
                               normalization='linear'
                               ):
        A = np.zeros((self.x.shape[0], self.x.shape[0]))
        if hasattr(self.model, 'A'):
            A = self.model.get_A(exp=exp)
            # A = (A-self.model.dt*torch.eye(A.shape[0]))
            title = 'matrix'
            if transition:
                dx = self.tester.dx
                weights = (dx.unsqueeze(1).numpy())/(dx.unsqueeze(0).numpy())
                A = A*weights
                title = 'transition '+title
            if inverted_trans:
                A = A.t()
                title = 'transposed correlation'
            if corr:
                A = (A@A.t())
                varis = torch.sqrt(A.diag()).unsqueeze(1)
                cross_varis = varis@varis.t()
                zero_mask = torch.logical_not(cross_varis.abs().bool())
                cross_varis[zero_mask] = 1.0
                A = A/cross_varis
                title = 'correlation'

        elif hasattr(self.model, 'feature'):
            feature = self.model.feature.detach()
            if self.model.activation is not None:
                feature = self.model.activation(feature)
            A = (feature.transpose(-1, -2)@feature).sum(0).numpy()
            title = 'kernel'
        else:
            raise Exception('Unknown model')

        fig = plt.figure()
        ax = fig.add_subplot()
        if normalization == 'linear':
            norm = Normalize(-1, 1)
        elif normalization == 'linear0':
            norm = Normalize(0, 1)
        elif normalization == 'log':
            norm = LogNorm(0.01, 1)
        elif normalization == 'symlog':
            norm = SymLogNorm(0.04, vmin=-1, vmax=1)
        elif 'none':
            norm = None
        else:
            raise Exception('specify a valid normalization type !')

        im = ax.pcolor(self.x,
                       self.x,
                       A,
                       cmap='jet',
                       # interpolation=None,
                       # extent=[self.x.min(),
                       #         self.x.max(),
                       #         self.x.max(),
                       #         self.x.min()],
                       norm=norm
                       )
        # ax.invert_xaxis()
        # ax.invert_yaxis()
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        ax.set_xlabel(r'$d$')
        ax.set_ylabel(r'$d$')
        ax.set_xticks([self.x[0], 0.5, 1.0, 2.0, self.x[-1]])
        ax.set_yticks([ 0.5, 1.0, 2.0, self.x[-1]])
        plt.colorbar(im)
        # ax.set_title('Model '+title)
        if show_colsum:
            fig = plt.figure()
            ax = fig.add_subplot()
            col_sum = A.sum(1)
            ax.plot(self.x, col_sum)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(r'$d$')
            ax.set_ylabel('Column sum')
            ax.set_title('Columns sums of A')

    def eigvals_from_model(self, transition=True, exp=True, show_inv=True):
        A = np.zeros((self.x.shape[0], self.x.shape[0]))
        if hasattr(self.model, 'A'):
            A = self.model.get_A(exp=exp)
            if transition:
                dx = self.tester.dx
                weights = (dx.unsqueeze(1).numpy())/(dx.unsqueeze(0).numpy())
                A = A*weights
        else:
            raise Exception('Model has no attribute A')
        eigs, eigvecs = np.linalg.eig(A)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(eigs.real,
                   eigs.imag)
        ax.set_xlabel('real')
        ax.set_ylabel('imaginary')
        ax.set_title('Model eigenvalues')

        if show_inv:
            idx = np.argmin(np.abs(eigs-1.0))
            invar_vec = -eigvecs[:, idx]/np.array(self.tester.dx)
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(self.x, invar_vec)
            ax.set_xlabel(r'$d$')
            ax.set_ylabel('weight')
            ax.set_title('Eigenvector associated to '+str(eigs[idx]))

    def correlation_from_data(self, normalization='log'):
        A = self.tester.correlation_from_data()
        fig = plt.figure()
        ax = fig.add_subplot()
        vmin = float(self.x.min())
        vmax = float(self.x.max())
        if normalization == 'linear':
            norm = Normalize(-1, 1)
        elif normalization == 'log':
            norm = LogNorm(0.01, 1)
        elif normalization == 'symlog':
            norm = SymLogNorm(0.04, vmin=-1, vmax=1)
        elif 'none':
            norm = None
        else:
            raise Exception('specify a valid normalization type !')

        # im = ax.imshow(A.T,
        #                cmap='jet',
        #                interpolation=None,
        #                extent=[vmin,
        #                        vmax,
        #                        vmax,
        #                        vmin],
        #                # origin='lower',
        #                norm=norm
        #                )
        im = ax.pcolor(self.x,
                       self.x,
                       A,
                       cmap='jet',
                       # interpolation=None,
                       # extent=[self.x.min(),
                       #         self.x.max(),
                       #         self.x.max(),
                       #         self.x.min()],
                       # origin='lower',
                       norm=norm
                       )
        # ax.invert_yaxis()
        plt.colorbar(im)
        ax.set_xlabel(r'$d$')
        ax.set_ylabel(r'$d$')
        # ax.set_title('Data autocorrelation')

    def error(self, one_step=True, relative=False):
        _, error_in_time, time_norm, error_per_diam, diam_norm = self.tester.generate_trajectory_from_model(
            one_step=one_step)
        fig = plt.figure()
        ax = fig.add_subplot()
        if relative:
            ax.plot(self.t, error_in_time/time_norm, label='l1 relative error')
        else:
            ax.plot(self.t, error_in_time, label='l1 error')
            ax.plot(self.t, time_norm, label='l1 norm of data')
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$l_1$ error')
        ax.set_yscale('log')
        ax.legend()
        ax.set_title('Error in time')

        fig = plt.figure()
        ax = fig.add_subplot()
        if relative:
            ax.plot(self.x, error_per_diam/diam_norm, label='l1 relative error')
        else:
            ax.plot(self.x, error_per_diam, label='l1 error')
            ax.plot(self.x, diam_norm, label='l1 norm of data')
        ax.set_xlabel(r'$d$')
        ax.set_ylabel(r'$l_1$ error')
        ax.set_yscale('log')
        ax.legend()
        ax.set_title('Error per diameter')

    def data_avg_t(self):
        avg_t = self.tester.average_in_space_data()
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(self.t, avg_t)
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$\overline{n}(t)$')
        ax.set_title('Average number density in time')

    def model_predict_avg_t(self):
        avg_t = self.tester.average_in_space_model_predict()
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(self.t, avg_t)
        ax.set_yscale('log')
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$\overline{n}(t)$')
        ax.set_title('Average predicted number density in time')

    def pairplot(self):
        if self.datadf is None:
            self.dataframe_from_data()
        pp = sns.pairplot(self.datadf,
                          y_vars=['value'],
                          diag_kind='kde')
        for ax in pp.axes.flat:
            if ax.get_xlabel() == 'd':
                ax.set(xscale='log')
                ax.set(yscale='log')

    def vars_corr(self):
        if self.datadf is None:
            self.dataframe_from_data()
        plt.figure()
        sns.heatmap(self.datadf.corr(),
                    annot=True,
                    cmap='coolwarm').set_title('Variables correlations')

    def value_kde(self):
        if self.datadf is None:
            self.dataframe_from_data()
        plt.figure()
        sns.kdeplot(data=self.datadf,
                    x="value"
                    ).set(title='Number density distribution',
                          xscale='log')

    @staticmethod
    def show():
        plt.show()
