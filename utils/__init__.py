<<<<<<< HEAD
from .rewards import sortino_ratio, sharpe_ratio, log_return, opportunity, opportunity_continuos
=======
from .rewards import sortino_ratio, sharpe_ratio, log_return, opportunity, opportunity_continuos, upper_lower, compare_baseline
>>>>>>> 3a230e5cd2b7bfb0609965529f37fb69a8899198
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