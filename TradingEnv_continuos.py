import gymnasium as gym
from gymnasium import spaces
import numpy as np
import utils

class TradingMarket(gym.Env):
    def __init__(self, env_config):
        super().__init__()

        self.data = {}
        self.historical_price = env_config["OHCL"]
        self.epi_len = env_config["epi_len"]
        self.initial_value = env_config["initial_value"]
        self.window = env_config["window"]

        obs, _ = self.reset()

        self.action_space = spaces.Box(0,1, dtype=np.float64)

        self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape = [len(obs)],
                dtype=np.float64
        )
        self.reward_func = utils.opportunity_continuos

    def valorisation(self, price):
        """
        Calcula qual o valor total do patrimônio em reais como preço da ação
        fornecido
        """
        return price*self.n_assets + self.avaiable_money

    def _get_obs(self):
        OHLC = self.historical_price.iloc[self.idx-self.window : self.idx, 3]

        # features
        MR = utils.mean_reversion(OHLC)
        RSI = utils.relative_strength_index(OHLC)
        MACD = utils.moving_average_convergence_divergence(OHLC)
        signal = utils.signal_MACD(OHLC[-34:])
        K = utils.slow_stochastic_oscillator(OHLC[-17:])
        self.position = 0
        observation = np.array([MR, RSI, MACD, signal, K, self.position])

        return observation

    def reset(self, seed=None, options=None):
        """
        Reset o ambiente ambiente para os valores inciais
        """
        np.random.seed(seed)
        self.idx = np.random.randint(max(self.window, 40), self.historical_price.shape[0] - self.epi_len)
        self.steps = 0
        self.position = 0
        self.avaiable_money = (1 - self.position) * self.initial_value
        price = self.historical_price.iloc[self.idx, :]['Close']
        self.n_assets = self.position * self.initial_value / price
        self.history = [(
            self.historical_price.iloc[0, :].name.strftime("%m/%d/%Y"),
            self.initial_value, 
            self.position,
            price
        )]
        self.done = False

        observation = self._get_obs()

        return observation, {}

    def step(self, action):
        """
        Executa uma ação no ambiente. Nesse caso, rebalancea o portifolio entre
        reais e ações compradas segundo a posição desejada
        """
        desired_position = action
        price = self.historical_price.iloc[self.idx, :]['Close']
        current_value = self.valorisation(price)

        if desired_position != self.position:
            ideal_n_assets = current_value*desired_position/price

            asset_to_trade = ideal_n_assets - self.n_assets

            operation_cost = asset_to_trade * price
            self.avaiable_money = self.avaiable_money - operation_cost
            self.n_assets = self.n_assets + asset_to_trade
            self.position = desired_position
            self.avaiable_money -= 0.001*asset_to_trade*price
        
        next_price = self.historical_price.iloc[self.idx+1, :]['Close']
        new_value = self.valorisation(next_price)

        self.history.append((
            self.historical_price.iloc[self.idx+1, :].name.strftime("%m/%d/%Y"), 
            new_value, 
            self.position,
            next_price
        ))

        self.idx += 1
        self.steps += 1
        reward = self.reward_func(
            self.history[max(self.steps-self.window, 0) : self.steps+1], 
            k=3
        )

        if self.steps == self.epi_len or self.idx == self.historical_price.shape[0]-1:
            self.done = True
        observation = self._get_obs()
        
        return observation, reward, self.done, False, {}

    def reward(self, history, **kwargs):
        """
        Função de recompensa que guiará o aprendizado.
        """
        up_low = utils.upper_lower(self.historical_price.iloc[self.idx: self.idx+2, :]['Close'], self.position)
        return 1*up_low