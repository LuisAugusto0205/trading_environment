<<<<<<< HEAD
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo.ppo import PPOConfig 

from tqdm import tqdm

from TsfreshEnv import TradingMarket, log_return
import os
import argparse
from models.convolutional import ConvModel, default_conv_model_config
from ray.rllib.models.catalog import ModelCatalog

import gymnasium as gym
import ray
from ray import air, tune

import numpy as np
import pandas_datareader.data as pdr
import yfinance
import utils

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

yfinance.pdr_override()

def run_experiement(
        config,
        max_steps,
        path='',
        exp=0
):
    print("load Train!")


    stop_criteria = {
        "timesteps_total": max_steps
    }

    # tuner = tune.Tuner(
    #     "PPO",
    #     param_space=config,
    #     run_config=air.RunConfig(
    #         stop=stop_criteria,
    #         verbose=2,
    #         checkpoint_config=air.CheckpointConfig(
    #             checkpoint_frequency=1, checkpoint_at_end=True
    #         ),
    #     ),
    # )
    # results = tuner.fit()

    agent_config = PPOConfig.from_dict(config)
    agent = agent_config.build()

    with open(f"{path}/eval_{exp}.txt", "w") as file:
        pass
    
    count_steps = 0
    #pbar = tqdm(total=max_steps)
    while count_steps < max_steps:
        info = agent.train()
        #pbar.update(10)
        count_steps = info['num_env_steps_trained']
        if count_steps % 20000 == 0:
            history, _ = evaluate(agent)

            results_patrimony = np.array([day[1] for day in history])
            results_date = np.array([day[0] for day in history])
            results_act = np.array([day[2] for day in history])

            with open(f"{path}/eval_{exp}.txt", "a") as file:
                file.write(f"{results_patrimony[-1]}\t{count_steps}\n")

    agent.stop()

    #checkpoint = results.get_best_result().checkpoint
    # Create new Algorithm and restore its state from the last checkpoint.
    #agent = Algorithm.from_checkpoint(checkpoint)

    return agent

def evaluate(agent):

    print("load Test!")

    env_eval = gym.make('TradeEnvTest-v0')
    obs, _ = env_eval.reset(seed=42)
    rewards = []
    done = False
    while not done:

        action = agent.compute_single_action(
                observation=obs,
                explore=False,
                policy_id="default_policy",  # <- default value
            )


        obs, rwd, done, *_ = env_eval.step(action)

        rewards.append(rwd)
    return env_eval.history, df_test


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ticket', type=str, default='BBAS3.SA',
                        help='Ticket from Yahoo Finance')
    parser.add_argument('--train_time', type=str, default='2000-01-01/2020-01-01',
                        help='Time range that agent will be trained')
    parser.add_argument('--eval_time', type=str, default='2020-01-01/2023-12-01',
                        help='Time range that agent will be evaluated')
    parser.add_argument('--patrimony', type=int, default=1000,
                        help='Patrimony that agent will start')
    parser.add_argument('--positions', nargs='+')
    parser.add_argument('--max_steps', type=int, default=200000,
                        help='stop criteria')
    parser.add_argument('--window_size', type=int, default=15,
                    help='length of each observation')
    parser.add_argument('--epi_len', type=int, default=300,
                    help='episode length')
    parser.add_argument('--n_times', type=int, default=1,
                        help='Repeat experiement N times')
    parser.add_argument('--model', type=str, default='linear',
                        help='model that will be used')
    

    args = parser.parse_args()

    custom_models = {
        "Conv": ConvModel
    }

    if args.model == 'FullyCon':
        model_config = {"fcnet_hiddens": [32, 16]}
    elif args.model == 'Conv':
        model_config = default_conv_model_config
        ModelCatalog.register_custom_model("ConvTorchModel", custom_models['Conv'])

    df_train = pdr.get_data_yahoo(args.ticket, args.train_time.split('/')[0], args.train_time.split('/')[1])
    df_test = pdr.get_data_yahoo(args.ticket, args.eval_time.split('/')[0], args.eval_time.split('/')[1])

    config_train = {
        "env": TradingMarket,
        "env_config": {
            "OHCL": df_train,
            "feat": utils.compute_features(df_train.reset_index(), args.window_size),
            "initial_value": args.patrimony,
            "positions": [float(p) for p in args.positions],
            "epi_len": args.epi_len,
            "window": args.window_size,
        },
        "num_workers": 7,
        "num_gpus":1,
        "framework":"torch",
        "model": model_config,
        # "_enable_learner_api": False,
        # "_enable_rl_module_api": False
    }

    gym.envs.register(
        id='TradeEnvTest-v0',
        entry_point=TradingMarket,
        kwargs={
            "env_config": {
                "OHCL": df_test,
                "feat": utils.compute_features(df_test.reset_index(), args.window_size),
                "initial_value": args.patrimony,
                "positions": [float(p) for p in args.positions], 
                "window": args.window_size,
                "epi_len":df_test.shape[0]-args.window_size-1
            }
        }
    )

    if not os.path.exists(f"results_test_ppo_{args.model}"):
        os.makedirs(f"results_test_ppo_{args.model}")

    results = []
    for i in range(args.n_times):
        ray.init()
        
        agent = run_experiement(
            config_train,
            max_steps=args.max_steps,
            path=f"results_test_ppo_{args.model}",
            exp=i
        )

        result, df_test = evaluate(agent)

        results.append(result)
        results_patrimony = np.array([day[1] for day in result])
        results_date = np.array([day[0] for day in result])
        results_act = np.array([day[2] for day in result])

        utils.plot_actions(
            results_act[1:], 
            results_date[1:], 
            df_test.iloc[-len(results_act[1:]):, :]['Close'], 
            f"results_test_ppo_{args.model}", 
            exp=i
        )
        ray.shutdown()
        
        with open(f"results_test_ppo_{args.model}/exp_{i}.txt", 'w') as file:
            file.write(f'{str(result)}')

    results_patrimony = np.array([ [day[1] for day in result] for result in results])
    results_date = [ [day[0] for day in result] for result in results]
    results_act = np.array([ [day[2] for day in result] for result in results])

    utils.plot_results(
        results_date[0][1:], 
        results_patrimony.mean(axis=0)[1:], 
        results_patrimony.std(axis=0)[1:],
        args.patrimony,
        args.model,
        f"results_test_ppo_{args.model}"
    )
=======
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm

from TradingEnv import TradingMarket, log_return
import os
import argparse
from models.convolutional import ConvModel, default_conv_model_config
from ray.rllib.models.catalog import ModelCatalog

import gymnasium as gym
import ray
from ray import air, tune

import numpy as np
import pandas_datareader.data as pdr
import yfinance
import utils

yfinance.pdr_override()

def run_experiement(
        ticket, 
        dt_inicial, 
        dt_final, 
        patrimony,
        positions,
        max_steps,
        window_size,
        episode_length,
        model_config={}
):

    df_train = pdr.get_data_yahoo(ticket, dt_inicial, dt_final)
    print("load Train!")

    config = {
        "env": TradingMarket,
        "env_config": {
            "data": df_train,
            "patrimony": patrimony,
            "positions": positions,
            "epi_len": episode_length,
            "window": window_size,
        },
        "num_workers": 7,
        "num_gpus":1,
        "framework":"torch",
        "model": model_config,
        "_enable_learner_api": False,
        "_enable_rl_module_api": False
    }

    stop_criteria = {
        "timesteps_total": max_steps
    }

    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=air.RunConfig(
            stop=stop_criteria,
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=1, checkpoint_at_end=True
            ),
        ),
    )
    results = tuner.fit()

    checkpoint = results.get_best_result().checkpoint
    # Create new Algorithm and restore its state from the last checkpoint.
    agent = Algorithm.from_checkpoint(checkpoint)

    return agent

def evaluate(
        agent,
        ticket, 
        dt_inicial, 
        dt_final, 
        patrimony,
        positions,
        window_size
):

    df_test = pdr.get_data_yahoo(ticket, dt_inicial, dt_final)
    print("load Test!")

    gym.envs.register(
        id='TradeEnv-v0',
        entry_point=TradingMarket,
        kwargs={
            "data": {"data":df_test},
            "initial_value": patrimony,
            "positions": positions,
            "reward_func": log_return, 
            "window":window_size,
            "dt_fim":"2023-11-29", 
            "epi_len":df_test.shape[0]-window_size-1
        }
    )

    env_eval = gym.make('TradeEnv-v0')
    obs, _ = env_eval.reset(seed=42)
    rewards = []
    done = False
    while not done:

        action = agent.compute_single_action(
                observation=obs,
                explore=False,
                policy_id="default_policy",  # <- default value
            )


        obs, rwd, done, *_ = env_eval.step(action)

        rewards.append(rwd)
    return env_eval.history, df_test


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ticket', type=str, default='BBAS3.SA',
                        help='Ticket from Yahoo Finance')
    parser.add_argument('--train_time', type=str, default='2000-01-01/2020-01-01',
                        help='Time range that agent will be trained')
    parser.add_argument('--eval_time', type=str, default='2020-01-01/2023-12-01',
                        help='Time range that agent will be evaluated')
    parser.add_argument('--patrimony', type=int, default=1000,
                        help='Patrimony that agent will start')
    parser.add_argument('--positions', nargs='+')
    parser.add_argument('--max_steps', type=int, default=200000,
                        help='stop criteria')
    parser.add_argument('--window_size', type=int, default=15,
                    help='length of each observation')
    parser.add_argument('--epi_len', type=int, default=300,
                    help='episode length')
    parser.add_argument('--n_times', type=int, default=1,
                        help='Repeat experiement N times')
    parser.add_argument('--model', type=str, default='linear',
                        help='model that will be used')
    

    args = parser.parse_args()

    custom_models = {
        "Conv": ConvModel
    }

    if not os.path.exists(f"results_test_ppo_{args.model}"):
        os.makedirs(f"results_test_ppo_{args.model}")

    results = []
    for i in range(args.n_times):
        ray.init()
        if args.model == 'FullyCon':
            model_config = {}#{"fcnet_hiddens": [64, 32]}
        elif args.model == 'Conv':
            model_config = default_conv_model_config
            ModelCatalog.register_custom_model("ConvTorchModel", custom_models['Conv'])
        
        agent = run_experiement(
            ticket=args.ticket, 
            dt_inicial=args.train_time.split('/')[0], 
            dt_final=args.train_time.split('/')[1], 
            patrimony=args.patrimony,
            positions=[float(pos) for pos in args.positions],
            max_steps=args.max_steps,
            window_size=args.window_size,
            episode_length=args.epi_len,
            model_config=model_config
        )

        result, df_test = evaluate(
                agent=agent,
                ticket=args.ticket, 
                dt_inicial=args.eval_time.split('/')[0], 
                dt_final=args.eval_time.split('/')[1], 
                patrimony=args.patrimony,
                positions=[float(pos) for pos in args.positions],
                window_size=args.window_size
        )

        results.append(result)
        results_patrimony = np.array([day[1] for day in result])
        results_date = np.array([day[0] for day in result])
        results_act = np.array([day[2] for day in result])

        utils.plot_actions(
            results_act[1:], 
            results_date[1:], 
            df_test.iloc[-len(results_act[1:]):, :]['Close'], 
            f"results_test_ppo_{args.model}", 
            exp=i
        )
        ray.shutdown()
        
        with open(f"results_test_ppo_{args.model}/exp_{i}.txt", 'w') as file:
            file.write(f'{str(result)}')

    results_patrimony = np.array([ [day[1] for day in result] for result in results])
    results_date = [ [day[0] for day in result] for result in results]
    results_act = np.array([ [day[2] for day in result] for result in results])

    utils.plot_results(
        results_date[0][1:], 
        results_patrimony.mean(axis=0)[1:], 
        results_patrimony.std(axis=0)[1:],
        args.patrimony,
        args.model,
        f"results_test_ppo_{args.model}"
    )
>>>>>>> 3a230e5cd2b7bfb0609965529f37fb69a8899198
