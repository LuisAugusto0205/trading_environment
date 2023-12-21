import numpy as np

def log_return(history):
    patrimony = [x[1] for x in history]
    return (patrimony[-1] - patrimony[-2])/patrimony[-2] #np.log(history[-1]/history[-2])

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