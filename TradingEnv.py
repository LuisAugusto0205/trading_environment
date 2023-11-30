import gymnasium as gym
from gymnasium import spaces
import numpy as np

def log_return(history):
    return np.log(history[-1]/history[-2])

def sharpe_ratio(history, risk_free=0):
    portifolio_return = np.mean(history)
    portifolio_std = np.std(history)
    return (portifolio_return - risk_free)/portifolio_std

def sortino_ratio(history, risk_free=0):
    portifolio_return = np.mean(history)
    portifolio_std = np.std([x for x in history if x < 0])
    portifolio_std = 1 if str(portifolio_std) == 'nan' else portifolio_std

    return (portifolio_return - risk_free)/portifolio_std

class TradingMarket(gym.Env):
    def __init__(self, data, initial_value=1000, positions=[1, 0], reward_func=log_return, window=15, dt_fim=None, epi_len=300):
        super().__init__()

        self.data = {}
        self.historical_price = data["data"]
        self.epi_len = epi_len
        self.positions = np.array(positions)
        self.initial_value = initial_value
        self.window = window
        self.idx = np.random.randint(self.window, self.historical_price.shape[0] - self.epi_len)
        self.steps = 0
        self.position = 0
        
        self.n_assets = self.position * self.initial_value / self.historical_price.iloc[self.idx, :]['Close']
        self.avaiable_money = (1 - self.position) * self.initial_value


        self.action_space = spaces.Discrete(len(positions))

        self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape = [self.window+2],
                dtype=np.float64
        )

        self.history = [initial_value]
        self.reward_func = reward_func
        self.done = False

    def valorisation(self, price):
        """
        Calcula qual o valor total do patrimônio em reais como preço da ação
        fornecido
        """
        return price*self.n_assets + self.avaiable_money

    def _get_obs(self):
        OHLC = self.historical_price.iloc[self.idx-self.window : self.idx, 3].values
        onehot_action = np.zeros((2), dtype=np.float64)
        onehot_action[np.where(self.positions == self.position)] = 1
        observation = np.concatenate([
            OHLC, 
            onehot_action
        ])
        return observation

    def reset(self, seed=None, options=None):
        """
        Reset o ambiente ambiente para os valores inciais
        """
        np.random.seed(seed)
        self.history = [self.initial_value]
        self.idx = np.random.randint(self.window, self.historical_price.shape[0] - self.epi_len)
        self.steps = 0
        self.done = False

        observation = self._get_obs()

        return observation, {}

    def step(self, desired_position):
        """
        Executa uma ação no ambiente. Nesse caso, rebalancea o portifolio entre
        reais e ações compradas segundo a posição desejada
        """
        price = self.historical_price.iloc[self.idx+1, :]['Close']
        current_value = self.valorisation(price)

        if desired_position != self.position:
            ideal_n_assets = current_value*desired_position/price

            asset_to_trade = ideal_n_assets - self.n_assets

            operation_cost = asset_to_trade * price
            self.avaiable_money = self.avaiable_money - operation_cost
            self.n_assets = self.n_assets + asset_to_trade
            self.position = desired_position

        self.history.append(current_value)

        self.idx += 1
        self.steps += 1
        reward = self.reward_func(
            self.history[
                max(self.steps-self.window, 0):
                self.steps+1
            ]
        )

        if self.steps == self.epi_len or self.idx == self.historical_price.shape[0]-1:
            self.done = True
        observation = self._get_obs()
        
        return observation, reward, self.done, False, {}

    def reward(self, history, **kwargs):
        """
        Função de recompensa que guiará o aprendizado.
        """
        return self.reward_func(history, **kwargs)