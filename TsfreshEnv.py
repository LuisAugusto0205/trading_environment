import gymnasium as gym
from gymnasium import spaces
import numpy as np

def log_return(history):
    return (history[-1] - history[-2])/history[-2] #np.log(history[-1]/history[-2])

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
    def __init__(self, env_config):
        super().__init__()

        self.data = {}
        self.historical_price = env_config["OHCL"]
        self.df_feat = env_config["feat"]
        self.epi_len = env_config["epi_len"]
        self.positions = np.array(env_config["positions"])
        self.initial_value = env_config["initial_value"]
        self.window = env_config["window"]
        self.idx = np.random.randint(0, self.df_feat.shape[0] - self.epi_len)

        self.steps = 0
        self.position = 0
        
        self.n_assets = self.position * self.initial_value / self.historical_price.iloc[self.idx, :]['Close']
        self.avaiable_money = (1 - self.position) * self.initial_value


        self.action_space = spaces.Discrete(len(self.positions))

    #     self.observation_space = spaces.Dict({
    #         "history_data": spaces.Box(
    #                 -np.inf,
    #                 np.inf,
    #                 shape = [self.window],
    #                 dtype=np.float64
    #         ),
    #         "position":spaces.Discrete(len(self.positions))
    #    })

        self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape = [self.df_feat.shape[1]+len(self.positions)],
                dtype=np.float64
        )

        self.history = [(
            self.historical_price.iloc[self.window, :].name.strftime("%m/%d/%Y"),
            self.initial_value, 
            self.position
        )]
        self.reward_func = log_return
        self.done = False

    def valorisation(self, price):
        """
        Calcula qual o valor total do patrimônio em reais como preço da ação
        fornecido
        """
        return price*self.n_assets + self.avaiable_money

    def _get_obs(self):
        feat = self.df_feat.iloc[self.idx, :].values
        onehot_action = np.zeros(self.positions.shape, dtype=np.float64)
        onehot_action[np.where(self.positions == self.position)] = 1
        observation = np.concatenate([
            feat, 
            onehot_action
        ])
        return observation

    def reset(self, seed=None, options=None):
        """
        Reset o ambiente ambiente para os valores inciais
        """
        np.random.seed(seed)
        self.idx = np.random.randint(0, self.df_feat.shape[0] - self.epi_len)
        self.steps = 0
        self.position = 0
        self.avaiable_money = (1 - self.position) * self.initial_value
        self.n_assets = self.position * self.initial_value / self.historical_price.iloc[self.idx, :]['Close']
        self.history = [(
            self.historical_price.iloc[0, :].name.strftime("%m/%d/%Y"),
            self.initial_value, 
            self.position
        )]
        self.done = False

        observation = self._get_obs()

        return observation, {}

    def step(self, action):
        """
        Executa uma ação no ambiente. Nesse caso, rebalancea o portifolio entre
        reais e ações compradas segundo a posição desejada
        """
        desired_position = self.positions[action]
        price = self.historical_price.iloc[self.idx+1, :]['Close']
        current_value = self.valorisation(price)

        if desired_position != self.position:
            ideal_n_assets = current_value*desired_position/price

            asset_to_trade = ideal_n_assets - self.n_assets

            operation_cost = asset_to_trade * price
            self.avaiable_money = self.avaiable_money - operation_cost
            self.n_assets = self.n_assets + asset_to_trade
            self.position = desired_position
            self.avaiable_money -= 0.001*asset_to_trade*price

        self.history.append((
            self.historical_price.iloc[self.idx+1, :].name.strftime("%m/%d/%Y"), 
            current_value, 
            self.position
        ))

        self.idx += 1
        self.steps += 1
        reward = self.reward_func([x[1] for x in
            self.history[
                max(self.steps-self.window, 0):
                self.steps+1
            ]
         ])

        if self.steps == self.epi_len or self.idx == self.historical_price.shape[0]-1:
            self.done = True
        observation = self._get_obs()
        
        return observation, reward, self.done, False, {}

    def reward(self, history, **kwargs):
        """
        Função de recompensa que guiará o aprendizado.
        """
        return self.reward_func(history, **kwargs)