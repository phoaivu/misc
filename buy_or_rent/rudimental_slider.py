import matplotlib.widgets as widgets


class RudimentalSlider(object):

    def __init__(self, slider_subplot, textbox_subplot, **kwargs):
        self._slider = widgets.Slider(slider_subplot, **kwargs)
        self._slider.valtext.set_visible(False)
        self._slider.on_changed(self._on_changed)
        self._slider_on_changed = None

        self._valbox = widgets.TextBox(textbox_subplot, label='', initial=self._slider.valtext.get_text())
        self._valbox.on_text_change(self._on_valbox_changed)

    def on_changed(self, func):
        self._slider_on_changed = func

    def _on_changed(self, slider_val):
        self._valbox.set_val(self._slider.valtext.get_text())
        if self._slider_on_changed is not None:
            self._slider_on_changed(slider_val)

    def _on_valbox_changed(self, new_val):
        try:
            numeric_val = float(new_val)
        except ValueError:
            self._valbox.set_val(self._slider.valtext.get_text())
            return

        self._slider.set_val(numeric_val)


'''
class PoorManLine(ParameterizedLine):

    def __init__(self, parameters, axes):
        super().__init__(parameters, axes, color='green', label='Poor man\'s')

    def calculate(self, x):
        beta_correction = 1 + 1. / self._parameters.beta
        vu_statistics = self._parameters.beta / (self._parameters.sigma - 1. - self._parameters.beta)
        upfront_payment = self._parameters.total_cost * (1. - self._parameters.gamma)

        accumulated_beta = np.power(1. + self._parameters.beta, x)
        powered_sigma = np.power(self._parameters.sigma, x)
        vals = (beta_correction * (upfront_payment * vu_statistics + self._parameters.partial_payment)) * accumulated_beta
        vals -= upfront_payment * beta_correction * vu_statistics * powered_sigma
        vals -= self._parameters.partial_payment * beta_correction
        return vals

    def refresh(self):
        if self._parameters.beta == 0 or self._parameters.sigma == 1 + self._parameters.beta:
            raise ValueError('Weeeeee!!!!!')
        x = np.arange(1, TIME_HORIZON, 1, dtype='float')
        self._line.set_xdata(x)
        self._line.set_ydata(self.calculate(x))
'''