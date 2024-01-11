# trading_environment
Repositório destinado a disciplina de residência do curso de bacharelado em inteligência artificial (INF-UFG). 
Um ambiente de simulação simplificado seguindo a interface do Gymnasium é implementado para aplicação de métodos de aprendizado por reforço.

## Treino
Para realizar um treinamento de algum algoritimo basta seguir a sintax da linha de comando

> python .\Algorithms\<Nome-do-algoritmo>\train.py 

Os algoritmos disponíveis são:
* A2C
* ARS
* DDPG
* DQN
* Q-Learning

Algumas flags podem ser passadas também para dizer qual a ação será utilizada (`--ticket`), o periodo de treino (`--train_time`) e número de episódios (`--n_eps`). Por exemplo, para treinar a DQN um possível comando seria:

> python .\Algorithms\DQN\train.py --ticket AAPL --train_time 2005-01-01/2020-01-01 --n_eps 6000

O comando acima irá treinar a DQN para ação da empresa apple de 2005 até 2020 por 6000 episódios

## Teste

Para o teste a seguinte sintaxe pode ser seguida:

> python .\Algorithms\<Nome-do-algoritmo>\test.py --save checkpoint-do-modelo

Os checkpoints são salvos na pasta results de cada algortimo, na flag `--save` deve ser passado o nome do arquivo pth ou pkl que está nessa pasta com os pesos do modelo salvo. Há também disponível as tags de `--ticket` e `--test-time`. Um exemplo com um checkpoint salvo da DQN pode ser:

> python .\Algorithms\DQN\test.py --ticket AAPL --test_time 2020-01-01/2024-01-01 --save dqnAgent_Trained_Model_AAPL-20240108-023442_up-low_best

Nesse exemplo, o checkpoint do DQN será testado nas ações da apple durante o periodo de 2020 a 2023

## Pacotes requisitados:
* ray[rllib]
* stable baselines
* strable baselines contrib
* pandas_datareader
* yfinance
