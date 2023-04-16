import functools
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

import rudimental_slider

TIME_HORIZON = 100
EPSILON = 1E-10

PARAMETER_SPECS = {
    'alpha': dict(label=r'Lai suat cho vay $\alpha$',
                  valmin=0.01, valmax=0.99, valinit=0.08, valstep=0.001, valfmt='%.5f'),
    'beta': dict(label=r'Truot gia  $\beta$',
                 valmin=0., valmax=0.99, valinit=0.02, valstep=0.001, valfmt='%.5f'),
    'sigma': dict(label=r'Lai suat tiet kiem $\sigma$',
                  valmin=0.01, valmax=0.99, valinit=0.07, valstep=0.001, valfmt='%.5f'),
    'gamma': dict(label=r'Ti le vay $\gamma$',
                  valmin=0.00, valmax=1, valinit=0.7, valstep=0.01, valfmt='%.5f'),
    'total_cost': dict(label=r'Tong gia tri $N$',
                       valmin=1000., valmax=30000, valinit=1700, valstep=100, valfmt='%d'),
    'partial_payment': dict(label='Tien thue $m$',
                            valmin=5. * 12, valmax=100 * 12, valinit=11 * 12, valstep=12, valfmt='%d'),
    'time_horizon': dict(label='Time horizon $T$', valmin=1, valmax=100, valinit=20, valstep=1, valfmt='%d')
}


class Parameters(object):

    alpha = 0.08
    beta = 0.02
    gamma = 0.7
    sigma = 0.07
    total_cost = 1700.
    partial_payment = 11. * 12
    time_horizon = 20

    names = ['alpha', 'beta', 'gamma', 'sigma', 'total_cost', 'partial_payment', 'time_horizon']

    def __init__(self, **kwargs):
        if any(k not in set(self.names) for k in kwargs.keys()):
            raise ValueError('Unexpected parameters')
        [setattr(self, n, v) for n, v in kwargs.items()]

    def sigma_correction(self):
        return 1. + 1./self.sigma

    def beta_correction(self):
        return 1. + 1./self.beta

    def accumulated_beta(self):
        return np.power(1. + self.beta, self.time_horizon)

    def accumulated_sigma(self):
        return np.power(1. + self.sigma, self.time_horizon)

    def copy(self, initializers=None):
        p = Parameters()
        for n in self.names:
            new_val = getattr(self, n) if initializers is None or n not in initializers else initializers[n]
            if not np.isscalar(new_val):
                new_val = np.copy(new_val)
            setattr(p, n, new_val)
        return p


def singularitize(func):
    def singularitized_func(params):
        n_scalars = sum(np.isscalar(getattr(params, n)) for n in params.names)
        if n_scalars == len(params.names):
            return func(params)

        if n_scalars < len(params.names) - 1:
            raise ValueError('Weeeeeeeeee')

        iterable_param_name = next(n for n in params.names if not np.isscalar(getattr(params, n)))
        return np.fromiter(map(lambda v: func(params.copy({iterable_param_name: v})),
                               getattr(params, iterable_param_name)), dtype='float')
    return singularitized_func


class Calculator(object):

    @staticmethod
    @singularitize
    def mortgage_cost(params):
        borrowing_money = params.total_cost * params.gamma

        if np.square(params.beta) <= EPSILON:
            return 0.5 * borrowing_money * params.alpha * (params.time_horizon + 1.)

        beta_correction = params.beta_correction()
        alpha_beta_ratio = 1. - params.alpha / params.beta

        periodic_mortgage_stats = (alpha_beta_ratio * beta_correction) / params.time_horizon
        accumulated_beta = params.accumulated_beta()

        vals = (borrowing_money * (-1. + params.alpha * beta_correction)) * accumulated_beta
        vals += borrowing_money * periodic_mortgage_stats * (accumulated_beta - 1.)
        return vals

    @staticmethod
    @singularitize
    def rent_cost(params):
        sigma_correction = params.sigma_correction()
        accumulated_sigma = params.accumulated_sigma()

        if np.square(params.beta) <= EPSILON:
            vals = params.partial_payment * params.time_horizon
            vals -= (params.total_cost * (1. - params.gamma) *
                     np.power(1. + params.sigma, params.time_horizon - 1.))
            return vals

        beta_correction = params.beta_correction()
        accumulated_beta = params.accumulated_beta()
        vals = params.partial_payment * beta_correction * (accumulated_beta - 1)

        if np.square(params.sigma - params.beta) <= EPSILON:
            vals -= (params.total_cost * (1. - params.gamma) *
                     (params.time_horizon * params.beta + 1.) *
                     np.power(1. + params.beta, params.time_horizon - 1.))
        else:
            correlation = (1. + params.beta) / (params.sigma - params.beta)
            vals -= (params.total_cost * (1. - params.gamma) * correlation *
                     (accumulated_sigma / sigma_correction - accumulated_beta / beta_correction))
        return vals

    @staticmethod
    @singularitize
    def buy_cost(params):
        return - params.total_cost * params.accumulated_beta()

    @staticmethod
    @singularitize
    def flipping_values(params):
        if np.square(params.beta) <= EPSILON:
            vals = 0.5 * params.gamma * params.alpha * (params.time_horizon + 1.)
            vals += (1. - params.gamma) * np.power(1. + params.sigma, params.time_horizon - 1.)
            vals /= params.time_horizon
            return vals

        accumulated_beta = params.accumulated_beta()
        alpha_beta_ratio = 1. - params.alpha / params.beta

        if np.square(params.sigma - params.beta) <= EPSILON:
            vals = params.gamma * (params.alpha - 1./params.beta_correction())
            vals += ((1. - params.gamma) * (params.beta * params.time_horizon + 1.) * params.beta /
                     np.square(1 + params.beta))
            vals *= (accumulated_beta / (accumulated_beta - 1))
            vals += params.gamma * alpha_beta_ratio / params.time_horizon
            return vals

        accumulated_sigma = params.accumulated_sigma()
        sigma_beta_ratio = params.beta / (params.sigma - params.beta)
        vals = (params.gamma * ((-1. / params.beta_correction()) + params.alpha +
                                (alpha_beta_ratio / params.time_horizon)) -
                (1. - params.gamma) * sigma_beta_ratio / params.beta_correction()) * accumulated_beta
        vals += accumulated_sigma * (1. - params.gamma) * sigma_beta_ratio / params.sigma_correction()
        vals -= params.gamma * alpha_beta_ratio / params.time_horizon
        vals /= (accumulated_beta - 1.)
        return vals


class ParameterizedLine(object):
    def __init__(self, axes, color='color', label='label', calculator=None):
        self._line = lines.Line2D([], [], color=color, label=label)
        self._calculator = calculator
        axes.add_line(self._line)

    def set_xdata(self, xdata):
        self._line.set_xdata(xdata)

    def refresh(self, params):
        self._line.set_ydata(self._calculator(params))


class MortgageLine(ParameterizedLine):
    def __init__(self, axes, color='red', label='Mortgage'):
        super().__init__(axes, color, label, calculator=Calculator.mortgage_cost)


class PoorManLine(ParameterizedLine):
    def __init__(self, axes, color='orange', label='Poor man\'s'):
        super().__init__(axes, color, label, calculator=Calculator.rent_cost)


class RichManLine(ParameterizedLine):
    def __init__(self, axes, color='violet', label='Rich man\'s'):
        super().__init__(axes, color, label, calculator=Calculator.buy_cost)


class FlippingLine(ParameterizedLine):
    def __init__(self, axes, color='green', label='m/N Flipping line'):
        super().__init__(axes, color, label, calculator=Calculator.flipping_values)


class Annotation(object):
    def __init__(self, axes):
        self._annotator = axes.annotate('wedge', xy=(2., -1), xycoords='data',
                                        xytext=(35, 0), textcoords='offset points',
                                        size=10, va="center", visible=False,
                                        bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"),
                                        arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                                                        fc=(1.0, 0.7, 0.7), ec="none",
                                                        patchA=None,
                                                        patchB=patches.Ellipse((2, -1), 0.5, 0.5),
                                                        relpos=(0.2, 0.5),
                                                        )
                                        )
        self._vertical_line = axes.axvline(0, color='black', linestyle='--', visible=False)
        self.current_xdata, self.current_ydata = None, None

    def update(self, x, y, text):
        xdata = x if x is not None else self.current_xdata
        ydata = y if y is not None else self.current_ydata
        if xdata is not None and ydata is not None:
            self._annotator.set_text(text)
            self._annotator.xy = (xdata, ydata)
            self._vertical_line.set_xdata([xdata, xdata])
            self.current_xdata = xdata
            self.current_ydata = ydata

    def set_visible(self, visible):
        self._annotator.set_visible(visible)
        self._vertical_line.set_visible(visible)
        if not visible:
            self.current_xdata, self.current_ydata = None, None


class SlidedDrawer(object):

    def __init__(self, fig, specs):
        self._fig = fig
        self._specs = specs
        self._params = Parameters()
        self._axes = None
        self._information = None
        self._annotator = None

    def setup(self):
        # add lines
        # unroll parameters, set axes limits
        raise NotImplementedError()

    def on_slider_changed(self, slider_val, param_name):
        raise NotImplementedError()

    def update_annotation(self, xdata, ydata):
        raise NotImplementedError()

    def draw(self):
        n_sliders = len(self._specs)
        main_gs = gridspec.GridSpec(3 * n_sliders + 1, 10, figure=self._fig)
        self._axes = self._fig.add_subplot(main_gs[:2 * n_sliders, :])
        plot_bounds = self._axes.get_position().bounds
        self._axes.set_position([plot_bounds[0], plot_bounds[1] + 0.1, plot_bounds[2], plot_bounds[3] - 0.03])

        information_axes = self._fig.add_subplot(main_gs[-1, :])
        information_axes.xaxis.set_visible(False)
        information_axes.yaxis.set_visible(False)
        self._information = information_axes.text(0, 0.5, '', horizontalalignment='left',
                                                  verticalalignment='center', transform=information_axes.transAxes)

        self._annotator = Annotation(self._axes)

        self.setup()

        def slider_changed(slider_val, param_name='param'):
            if not hasattr(self._params, param_name):
                raise ValueError('Invalid parameter name: {}'.format(param_name))

            setattr(self._params, param_name, slider_val)
            self.on_slider_changed(slider_val, param_name)
            self.update_annotation(xdata=None, ydata=None)
            self._fig.canvas.draw()

        slider_axes = set()
        for i, name in enumerate(sorted(self._specs.keys())):
            spec = self._specs[name]
            ax1 = self._fig.add_subplot(main_gs[2 * n_sliders + i, :-2])
            ax2 = self._fig.add_subplot(main_gs[2 * n_sliders + i, -2:])
            slider = rudimental_slider.RudimentalSlider(ax1, ax2,
                                                        label=spec['label'],
                                                        valmin=spec['valmin'], valmax=spec['valmax'],
                                                        valinit=spec['valinit'], valstep=spec['valstep'],
                                                        valfmt=spec['valfmt'])
            slider_axes.add(ax1)
            slider_axes.add(ax2)

            slider.on_changed(functools.partial(slider_changed, param_name=name))
            slider_changed(slider_val=spec['valinit'], param_name=name)

        def on_mouse_clicked(event_data):
            if event_data.inaxes == self._axes:
                self.update_annotation(event_data.xdata, event_data.ydata)
                self._annotator.set_visible(True)
            elif event_data.inaxes not in slider_axes:
                self._annotator.set_visible(False)
            self._fig.canvas.draw()

        self._fig.canvas.mpl_connect('button_release_event', on_mouse_clicked)


class CostDrawer(SlidedDrawer):
    def __init__(self, fig):
        specs = PARAMETER_SPECS.copy()
        specs.pop('time_horizon')
        super().__init__(fig, specs)
        self._drawn_lines = []

    def setup(self):
        self._drawn_lines = [MortgageLine(self._axes),
                             PoorManLine(self._axes),
                             RichManLine(self._axes)]
        self._params.time_horizon = np.arange(1, TIME_HORIZON, 1, dtype='float')
        [line.set_xdata(np.copy(self._params.time_horizon)) for line in self._drawn_lines]

        self._axes.set_title('Cost over time horizon')
        self._axes.set_ylabel('Cost (millions)')
        self._axes.set_xlabel('Time horizon $T$')
        self._axes.set_ylim(-1E4, 1E4)
        self._axes.set_xlim(0, TIME_HORIZON)
        self._axes.legend()
        self._axes.grid(True)
        [self._axes.axvline(x=x, color='blue', linestyle='--') for x in [10, 15, 20, 30]]

    def on_slider_changed(self, slider_val, param_name):
        self._information.set_text(r'm/N = {:.5f}; N*(1-$\gamma$) = {:.5f}'.format(
            self._params.partial_payment / self._params.total_cost,
            self._params.total_cost * (1. - self._params.gamma)))

        for line in self._drawn_lines:
            line.refresh(self._params)

    def update_annotation(self, xdata, ydata):
        params = self._params.copy()
        params.time_horizon = xdata if xdata is not None else self._annotator.current_xdata
        if params.time_horizon is not None:
            values = [('Mortgage', Calculator.mortgage_cost),
                      ('Poor man', Calculator.rent_cost),
                      ('Rich man', Calculator.buy_cost)]
            annotation_text = 'T = {:.5f}\n\n{}'.format(
                params.time_horizon,
                '\n'.join(r'{}: {:.5f}'.format(v[0], v[1](params)) for v in values))
            self._annotator.update(xdata, ydata, annotation_text)


class FlippingLineDrawer(SlidedDrawer):
    def __init__(self, fig):
        specs = PARAMETER_SPECS.copy()
        specs.pop('total_cost')
        specs.pop('partial_payment')
        specs.pop('time_horizon')
        super().__init__(fig, specs)
        self._flip_line = None

    def setup(self):
        self._flip_line = FlippingLine(self._axes)
        self._params.time_horizon = np.arange(1, TIME_HORIZON, 1, dtype='float')
        self._flip_line.set_xdata(np.copy(self._params.time_horizon))

        self._axes.set_title('Flipping value over Time horizon')
        self._axes.set_ylabel(r'$\frac{-b}{A}$')
        self._axes.set_xlabel('Time horizon $T$')
        self._axes.set_ylim(0, 1)
        self._axes.set_xlim(0, TIME_HORIZON)
        self._axes.legend()
        self._axes.grid(True)
        [self._axes.axvline(x=x, color='blue', linestyle='--') for x in [10, 15, 20, 30]]

    def on_slider_changed(self, slider_val, param_name):
        self._flip_line.refresh(self._params)

    def update_annotation(self, xdata, ydata):
        params = self._params.copy()
        params.time_horizon = xdata if xdata is not None else self._annotator.current_xdata
        if params.time_horizon is not None:
            params.time_horizon = np.round(params.time_horizon, 0)
            annotation_text = 'T = {:.5f}\n\nFlipping value: {:.5f}'.format(
                params.time_horizon, Calculator.flipping_values(params))
            self._annotator.update(params.time_horizon, ydata, annotation_text)


class CostOverBetaDrawer(SlidedDrawer):
    def __init__(self, fig):
        specs = PARAMETER_SPECS.copy()
        specs.pop('beta')
        super().__init__(fig, specs)
        self._drawn_lines = []

    def setup(self):
        self._drawn_lines = [MortgageLine(self._axes),
                             PoorManLine(self._axes),
                             RichManLine(self._axes)]
        self._params.beta = np.arange(-1 + 0.005, 1., 0.01, dtype='float')
        [line.set_xdata(np.copy(self._params.beta)) for line in self._drawn_lines]

        self._axes.set_title('Cost over beta')
        self._axes.set_ylabel('Cost (millions)')
        self._axes.set_xlabel(r'$\beta$')
        self._axes.set_ylim(-1E4, 1E4)
        self._axes.set_xlim(-1, 1.)
        self._axes.legend()
        self._axes.grid(True)
        [self._axes.axvline(x=x, color='blue', linestyle='--') for x in [0.02, 0.07]]

    def on_slider_changed(self, slider_val, param_name):
        for line in self._drawn_lines:
            line.refresh(self._params)

    def update_annotation(self, xdata, ydata):
        params = self._params.copy()
        params.beta = xdata if xdata is not None else self._annotator.current_xdata
        if params.beta is not None:
            values = [('Mortgage', Calculator.mortgage_cost),
                      ('Poor man', Calculator.rent_cost),
                      ('Rich man', Calculator.buy_cost)]
            annotation_text = '$\\beta$ = {:.5f}\n\n{}'.format(
                params.beta,
                '\n'.join(r'{}: {:.5f}'.format(v[0], v[1](params)) for v in values))
            self._annotator.update(xdata, ydata, annotation_text)


class FlippingLineOverBetaDrawer(SlidedDrawer):

    def __init__(self, fig):
        specs = PARAMETER_SPECS.copy()
        specs.pop('total_cost')
        specs.pop('partial_payment')
        specs.pop('beta')
        super().__init__(fig, specs)
        self._flip_line = None

    def setup(self):
        self._flip_line = FlippingLine(self._axes)
        self._params.beta = np.arange(-1 + 0.005, 1., 0.01, dtype='float')
        self._flip_line.set_xdata(np.copy(self._params.beta))

        self._axes.set_title('Flipping value over $\\beta$')
        self._axes.set_ylabel(r'$\frac{-b}{A}$')
        self._axes.set_xlabel('$\\beta$')
        self._axes.set_ylim(-2, 2)
        self._axes.set_xlim(-1, 1)
        self._axes.legend()
        self._axes.grid(True)
        [self._axes.axvline(x=x, color='blue', linestyle='--') for x in [0.02, 0.07]]

    def on_slider_changed(self, slider_val, param_name):
        self._flip_line.refresh(self._params)

    def update_annotation(self, xdata, ydata):
        params = self._params.copy()
        params.beta = xdata if xdata is not None else self._annotator.current_xdata
        if params.beta is not None:
            annotation_text = '$\\beta$ = {:.5f}\n\nFlipping value: {:.5f}'.format(
                params.beta, Calculator.flipping_values(params))
            self._annotator.update(xdata, ydata, annotation_text)


def main():
    plt.ion()

    # CostDrawer(plt.figure()).draw()
    FlippingLineDrawer(plt.figure()).draw()
    # CostOverBetaDrawer(plt.figure()).draw()
    # FlippingLineOverBetaDrawer(plt.figure()).draw()

    plt.show()
    input('')


if __name__ == '__main__':
    main()
