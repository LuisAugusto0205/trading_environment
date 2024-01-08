import numpy as np

def log_return(history, k=7):
    patrimony = [x[1] for x in history[-k-1:]]
    daily_return = (patrimony[1:] - patrimony[:-1])/patrimony[:-1]
    return np.tanh(daily_return.mean()) #np.log(history[-1]/history[-2])

def sharpe_ratio(history, risk_free=0):
    portifolio_return = np.mean(history)
    portifolio_std = np.std(history)
    return (portifolio_return - risk_free)/portifolio_std

def sortino_ratio(history, risk_free=0):
    portifolio_return = np.mean(history)
    portifolio_std = np.std([x for x in history if x < 0])
    portifolio_std = 1 if str(portifolio_std) == 'nan' else portifolio_std

    return (portifolio_return - risk_free)/portifolio_std

def opportunity(history, k=2):
    if len(history) < k:
        return 0
    prices = [h[3] for h in history]
    action = history[-1][2]
    bullish = True
    for i in range(k, 1, -1):
        bullish = bullish and prices[-i] < prices[-i+1]

    bearing = True
    for i in range(k, 1, -1):
        bearing = bearing and prices[-i] > prices[-i+1]

    if bullish and action==1:
        return 1
    elif bullish and action==0:
        return -1
    elif bearing and action==1:
        return -1
    elif bearing and action==0:
        return 1
    else:
        return 0

def upper_lower(prices, action):
    if prices[0] > prices[1]:
        if action == 0:
            return 1
        elif action == 1:
            return -1
    else:
        if action == 0:
            return -1
        elif action == 1:
            return 1

def compare_baseline(history, valorisation, k=15):
    last_price = history[-1][3]
    initial_price = history[-k][3]
    initial_patrimony = history[-k][1]

    baseline = last_price*(initial_patrimony/initial_price)
    return np.tanh(valorisation/baseline - 1)


def opportunity_continuos(history, k=2):
    if len(history) < k:
        return 0
    prices = [h[3] for h in history]
    action = history[-1][2]
    bullish = True
    for i in range(k, 1, -1):
        bullish = bullish and prices[-i] < prices[-i+1]

    bearing = True
    for i in range(k, 1, -1):
        bearing = bearing and prices[-i] > prices[-i+1]

    if bullish and action>=0.5:
        return 1
    elif bullish and action<0.5:
        return -1
    elif bearing and action>=0.5:
        return -1
    elif bearing and action<0.5:
        return 1
    else:
        return 0