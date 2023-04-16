import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

import buy_or_rent

FIGURE_OUTPUT_PATH = r'C:\Data\writing\Non fiction\TraGopHayThueNha\figures'


class TestBuyOrRent(unittest.TestCase):

    def test_banks(self):
        params = buy_or_rent.Parameters(
            alpha=0.0743,  # 0.075
            sigma=0.075,  # 0.072
            beta=0.0691,
            gamma=0.65,
            time_horizon=20,
            partial_payment=11. * 12,
            total_cost=1700.)
        print(buy_or_rent.Calculator.flipping_values(params))
        delta = buy_or_rent.Calculator.mortgage_cost(params) - buy_or_rent.Calculator.rent_cost(params)
        print(delta)
        print(delta / params.accumulated_beta())

    def test_s_by_t_varied_beta(self):
        params = buy_or_rent.Parameters(
            alpha=0.08,
            sigma=0.07,
            beta=0.02,
            gamma=0.7,
            total_cost=1700.,
            partial_payment=11. * 12,
            time_horizon=np.arange(1, 100, dtype='float'))

        with plt.xkcd(randomness=1):
            fig = plt.figure(figsize=(7, 3))
            axes = fig.add_subplot(1, 1, 1)
            for beta, color in zip([0, 0.04, 0.07, 0.1, 0.15], ['red', 'green', 'blue', 'violet', 'black']):
                params.beta = beta
                axes.add_line(lines.Line2D(params.time_horizon, buy_or_rent.Calculator.mortgage_cost(params),
                                           color=color,
                                           label='$\\beta$={:.2f}'.format(params.beta),
                                           linewidth=2))
            # fig.suptitle('Mua tra góp(S)')
            axes.set_title('$\\alpha={:.2f}, \\gamma={:.2f}, '
                           '\\frac{{m}}{{N}}={:.3f}$'.format(
                params.alpha, params.gamma, params.partial_payment / params.total_cost),
                fontdict=dict(fontsize=10, fontweight=1.))
            plot_bounds = axes.get_position().bounds
            axes.set_position([plot_bounds[0] + 0.02, plot_bounds[1] + 0.1, plot_bounds[2], plot_bounds[3] - 0.1])
            axes.set_ylim(-5E3, 1.5E4)
            axes.set_xlim(0, 100)
            axes.set_xticks([0, 10, 15, 20, 30, 40, 60, 80, 100])
            axes.set_xlabel('T (năm)')
            axes.set_ylabel('S')
            axes.legend()
            axes.grid(True)
            axes.axhline(y=0, color='blue', linestyle='--', linewidth=0.9)
            [axes.axvline(x=x, color='blue', linestyle='--', linewidth=0.9) for x in [10, 15, 20, 30]]
            plt.savefig(os.path.join(FIGURE_OUTPUT_PATH, 's_by_t_varied_beta.svg'))

    def test_r_by_t_varied_beta(self):
        params = buy_or_rent.Parameters(
            alpha=0.08,
            sigma=0.07,
            beta=0.02,
            gamma=0.7,
            total_cost=1700.,
            partial_payment=11. * 12,
            time_horizon=np.arange(1, 100, dtype='float'))

        with plt.xkcd(randomness=1):
            fig = plt.figure(figsize=(7, 3))
            axes = fig.add_subplot(1, 1, 1)
            for beta, color in zip([0, 0.04, 0.07, 0.1, 0.15], ['red', 'green', 'blue', 'violet', 'black']):
                params.beta = beta
                axes.add_line(lines.Line2D(params.time_horizon, buy_or_rent.Calculator.rent_cost(params),
                                           color=color,
                                           label='$\\beta$={:.2f}'.format(params.beta),
                                           linewidth=2))
            # fig.suptitle('Thuê (R)')
            axes.set_title('$\\sigma={:.2f}, \\gamma={:.2f}, '
                           '\\frac{{m}}{{N}}={:.3f}$'.format(
                params.sigma, params.gamma, params.partial_payment / params.total_cost),
                fontdict=dict(fontsize=10, fontweight=1.))
            plot_bounds = axes.get_position().bounds
            axes.set_position([plot_bounds[0] + 0.02, plot_bounds[1] + 0.1, plot_bounds[2], plot_bounds[3] - 0.1])
            axes.set_ylim(-5E3, 1.5E4)
            axes.set_xlim(0, 100)
            axes.set_xticks([0, 10, 15, 20, 30, 40, 60, 80, 100])
            axes.set_xlabel('T (năm)')
            axes.set_ylabel('R')
            axes.legend()
            axes.grid(True)
            axes.axhline(y=0, color='blue', linestyle='--', linewidth=0.9)
            [axes.axvline(x=x, color='blue', linestyle='--', linewidth=0.9) for x in [10, 15, 20, 30]]
            plt.savefig(os.path.join(FIGURE_OUTPUT_PATH, 'r_by_t_varied_beta.svg'))

    def test_flip_value_by_t_varied_beta(self):
        params = buy_or_rent.Parameters(
            alpha=0.08,
            sigma=0.07,
            beta=0.02,
            gamma=0.7,
            total_cost=1700.,
            partial_payment=11. * 12,
            time_horizon=np.arange(1, 100, dtype='float'))

        with plt.xkcd(randomness=1):
            fig = plt.figure(figsize=(7, 3))
            axes = fig.add_subplot(1, 1, 1)
            for beta, color in zip([0, 0.04, 0.07, 0.1, 0.15], ['red', 'green', 'blue', 'violet', 'black']):
                params.beta = beta
                axes.add_line(lines.Line2D(params.time_horizon, buy_or_rent.Calculator.flipping_values(params),
                                           color=color,
                                           label='$\\beta$={:.2f}'.format(params.beta),
                                           linewidth=2))
            # fig.suptitle('$\\frac{{-b}}{{A}}$')
            axes.set_title('$\\alpha={:.2f}, \\sigma={:.2f}, \\gamma={:.2f}, '
                           '\\frac{{m}}{{N}}={:.3f}$'.format(
                params.alpha, params.sigma, params.gamma, params.partial_payment / params.total_cost),
                fontdict=dict(fontsize=10, fontweight=1.))
            plot_bounds = axes.get_position().bounds
            axes.set_position([plot_bounds[0] + 0.02, plot_bounds[1] + 0.1, plot_bounds[2], plot_bounds[3] - 0.1])
            axes.set_autoscaley_on(True)
            axes.set_xlim(0, 100)
            axes.set_xticks([0, 10, 15, 20, 30, 40, 60, 80, 100])
            axes.set_xlabel('T (năm)')
            axes.set_ylabel('$\\frac{{-b}}{{A}}$')
            axes.legend()
            axes.grid(True)
            axes.axhline(y=0, color='blue', linestyle='--', linewidth=0.9)
            [axes.axvline(x=x, color='blue', linestyle='--', linewidth=0.9) for x in [10, 15, 20, 30]]
            plt.savefig(os.path.join(FIGURE_OUTPUT_PATH, 'flip_value_by_t_varied_beta.svg'))

    def test_s_by_beta_varied_t(self):
        params = buy_or_rent.Parameters(
            alpha=0.08,
            sigma=0.07,
            beta=np.arange(-0.99, 1, 0.01, dtype='float'),
            gamma=0.65,
            total_cost=1700.,
            partial_payment=11. * 12,
            time_horizon=10)

        with plt.xkcd(randomness=1):
            fig = plt.figure(figsize=(7, 3))
            axes = fig.add_subplot(1, 1, 1)
            axes.axhline(y=0, color='blue', linestyle='--', linewidth=0.9)
            axes.axvline(x=0, color='blue', linestyle='--', linewidth=0.9)

            for time_horizon, color in zip([1, 10, 15, 30, 50], ['red', 'green', 'blue', 'violet', 'black']):
                params.time_horizon = time_horizon
                axes.add_line(lines.Line2D(params.beta, buy_or_rent.Calculator.mortgage_cost(params),
                                           color=color,
                                           label='$T$={:d}'.format(params.time_horizon),
                                           linewidth=2))
            # fig.suptitle('Mua tra góp(S)')
            axes.set_title('$\\alpha={:.2f}, \\gamma={:.2f}, '
                           '\\frac{{m}}{{N}}={:.3f}$'.format(
                params.alpha, params.gamma, params.partial_payment / params.total_cost),
                fontdict=dict(fontsize=10, fontweight=1.))
            plot_bounds = axes.get_position().bounds
            axes.set_position([plot_bounds[0] + 0.04, plot_bounds[1] + 0.1, plot_bounds[2], plot_bounds[3] - 0.1])
            axes.set_ylim(-2E4, 4E4)
            axes.set_xlim(-0.99, 1)
            # axes.set_xticks([-0.99, 0, 0.02, 0.07, 1])
            axes.set_xlabel('$\\beta$')
            axes.set_ylabel('S')
            axes.legend()
            axes.grid(True)
            # [axes.axvline(x=x, color='blue', linestyle='--', linewidth=0.9) for x in [0.07]]
            plt.savefig(os.path.join(FIGURE_OUTPUT_PATH, 's_by_beta_varied_t.svg'))

    def test_r_by_beta_varied_t(self):
        params = buy_or_rent.Parameters(
            alpha=0.08,
            sigma=0.07,
            beta=np.arange(-0.99, 1, 0.01, dtype='float'),
            gamma=0.65,
            total_cost=1700.,
            partial_payment=11. * 12,
            time_horizon=20)

        with plt.xkcd(randomness=1):
            fig = plt.figure(figsize=(7, 3))
            axes = fig.add_subplot(1, 1, 1)
            axes.axhline(y=0, color='blue', linestyle='--', linewidth=0.9)
            axes.axvline(x=0, color='blue', linestyle='--', linewidth=0.9)

            for time_horizon, color in zip([1, 10, 15, 30, 50], ['red', 'green', 'blue', 'violet', 'black']):
                params.time_horizon = time_horizon
                axes.add_line(lines.Line2D(params.beta, buy_or_rent.Calculator.rent_cost(params),
                                           color=color,
                                           label='$T$={:d}'.format(params.time_horizon),
                                           linewidth=2))
            # fig.suptitle('Thuê (R)')
            axes.set_title('$\\sigma={:.2f}, \\gamma={:.2f}, '
                           '\\frac{{m}}{{N}}={:.3f}$'.format(
                params.sigma, params.gamma, params.partial_payment / params.total_cost),
                fontdict=dict(fontsize=10, fontweight=1.))
            plot_bounds = axes.get_position().bounds
            axes.set_position([plot_bounds[0] + 0.04, plot_bounds[1] + 0.1, plot_bounds[2], plot_bounds[3] - 0.1])
            axes.set_ylim(-2E4, 4E4)
            axes.set_xlim(-0.99, 1)
            # axes.set_xticks([-0.99, 0, 0.02, 0.07, 1, 2, 3])
            axes.set_xlabel('$\\beta$')
            axes.set_ylabel('R')
            axes.legend()
            axes.grid(True)
            plt.savefig(os.path.join(FIGURE_OUTPUT_PATH, 'r_by_beta_varied_t.svg'))

    def test_flip_value_by_beta_varied_t(self):
        params = buy_or_rent.Parameters(
            alpha=0.08,
            sigma=0.07,
            beta=np.arange(-0.99, 1, 0.01, dtype='float'),
            gamma=0.65,
            total_cost=1700.,
            partial_payment=11. * 12,
            time_horizon=20)

        with plt.xkcd(randomness=1):
            fig = plt.figure(figsize=(7, 3))
            axes = fig.add_subplot(1, 1, 1)
            for time_horizon, color in zip([1, 10, 15, 30, 50], ['red', 'green', 'blue', 'violet', 'black']):
                params.time_horizon = time_horizon
                axes.add_line(lines.Line2D(params.beta, buy_or_rent.Calculator.flipping_values(params),
                                           color=color,
                                           label='$T$={:d}'.format(params.time_horizon),
                                           linewidth=2))
            # fig.suptitle('$\\frac{{-b}}{{A}}$')
            axes.set_title('$\\alpha={:.2f}, \\sigma={:.2f}, \\gamma={:.2f}, '
                           '\\frac{{m}}{{N}}={:.3f}$'.format(
                params.alpha, params.sigma, params.gamma, params.partial_payment / params.total_cost),
                fontdict=dict(fontsize=10, fontweight=1.))
            plot_bounds = axes.get_position().bounds
            axes.set_position([plot_bounds[0] + 0.02, plot_bounds[1] + 0.1, plot_bounds[2], plot_bounds[3] - 0.1])
            axes.set_ylim(-0.5, 0.7)
            axes.set_xlim(-0.99, 1)
            # axes.set_xticks([-0.99, 0, 0.02, 0.07, 1, 2, 3])
            axes.set_xlabel('$\\beta$')
            axes.set_ylabel('$\\frac{{-b}}{{A}}$')
            axes.legend()
            axes.grid(True)
            axes.axhline(y=0, color='blue', linestyle='--', linewidth=0.9)
            axes.axvline(x=0, color='blue', linestyle='--', linewidth=0.9)
            plt.savefig(os.path.join(FIGURE_OUTPUT_PATH, 'flip_value_by_beta_varied_t.svg'))


if __name__ == '__main__':
    unittest.main()
