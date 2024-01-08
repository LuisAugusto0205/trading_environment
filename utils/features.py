import numpy as np

def EMA(data, a):
    weigths = np.array([(a if i != len(data)-1 else 1)*(1-a)**i for i in range(len(data))[::-1]])
    exponetial_moving_averange = np.sum(np.multiply(data, weigths))
    return exponetial_moving_averange

def mean_reversion(data):
    center_mean = data - data.mean()
    std = (data - data.mean()).std()
    MR = center_mean[-1]/std
    return MR

def relative_strength_index(data):
    ret = data - data.shift(1)
    gain = np.array([ 0 if x < 0 else x for x in ret[1:]])
    loss = np.array([ 0 if x > 0 else x for x in ret[1:]])
    average_gain = EMA(gain, a=1/len(gain))
    average_loss = EMA(abs(loss), a=1/len(gain))
    RSI = 100 - (100/ (1 + average_gain/average_loss))
    return RSI

def moving_average_convergence_divergence(data):
    ema12 = EMA(data[-12:], a=2/13)
    ema26 = EMA(data[-26:], a=2/27)
    MACD = ema12 - ema26
    return MACD

def signal_MACD(data):
    MACDs = []
    for i in range(9):
        inf = -35+i+1
        sup = -9+i+1
        sup = sup if sup < 0 else None
        MACDs.append(
            moving_average_convergence_divergence(data[inf:sup])
        )
    signal = EMA(MACDs, a=2/10)
    return signal

def fast_stochastic_oscillator(data):
    K = 100 * (data[-1] - data.min()) / (data.max() - data.min())
    return K

def slow_stochastic_oscillator(data):
    fast_k = np.zeros(3)
    for i in range(3):
        inf = -17+i+1
        sup = -3+i+1
        sup = sup if sup < 0 else None
        fast_k[i] = fast_stochastic_oscillator(data[inf:sup])
    return fast_k.mean()