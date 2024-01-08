from .rewards import sortino_ratio, sharpe_ratio, log_return, opportunity, opportunity_continuos
from .graphs import plot_actions, plot_results
from .features import (
    EMA, 
    mean_reversion, 
    relative_strength_index, 
    moving_average_convergence_divergence,
    signal_MACD,
    slow_stochastic_oscillator
)
from .others import evaluate