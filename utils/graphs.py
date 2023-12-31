import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import OrderedDict
import os

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

def plot_results(x, value_mean, value_std, initial_value, model, path, algo='PPO'):

    plt.figure(figsize=(20, 8))
    plt.title(f"Desempenho do {algo}_{model} nos dados de teste", fontsize=24, fontweight="bold", loc='left')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Tempo em Dias", fontsize=18, loc='left')
    plt.ylabel("Valor da carteira de investimentos (R$)", fontsize=18, loc='top')

    plt.plot(x, value_mean, label='PPO', color='green')
    plt.text(x[-1], value_mean[-1]+10, f'{value_mean[-1]:.2f}', fontsize=18, color='green')
    plt.fill_between(x, value_mean-value_std, value_mean+value_std ,alpha=0.3, facecolor='green')

    plt.hlines(initial_value, x[0], x[-1], color='black', linestyles='--', label='Patrimônio inicial')

    plt.xticks(list(range(len(x)))[::50], x[::50], rotation=45)
    plt.tight_layout()
    plt.legend(prop={"size":16}, frameon=False)
    if not os.path.exists(f'{path}/graphics'):
        os.makedirs(f'{path}/graphics')
    plt.savefig(f'{path}/graphics/test_result_{algo}_{model}.jpeg')

def plot_actions(acts, dates, prices, path, exp):
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_title("Comportamento do agente ao longo do tempo no período de avaliação", fontsize=24)
    ax.set_ylabel("Preço da ação", fontsize=18)
    ax.set_xlabel("Tempo em dias", fontsize=18)
    for i, (start, stop) in enumerate(zip(prices[:-1], prices[1:])):
        if acts[i] == 0:
            ax.plot([dates[i], dates[i+1]], [start, stop], color='red', label=f'0% Investido')
        elif acts[i] == 1:
            ax.plot([dates[i], dates[i+1]], [start, stop], color='blue', label='100% Investido')

    ax.set_xticks(list(range(len(dates)))[::50], dates[::50], rotation = 45)

    if not os.path.exists(f'{path}/graphics'):
        os.makedirs(f'{path}/graphics')
        
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), prop={"size":16})

    plt.tight_layout()
    fig.savefig(f'{path}/graphics/acts_{exp}.jpeg')