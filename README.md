# trading_environment
Repositório destinado a disciplina de residência do curso de bacharelado em inteligência artificial (INF-UFG). 
Um ambiente de simulação simplificado seguindo a interface do Gymnasium é implementado para aplicação de métodos de aprendizado por reforço.

Para configurar um experimento, agora é possível customizar os parâmetros do treino por meio da linha comando que são:
* **ticket**: Representa o nome do ativo registrado na plataforma Yahoo Finance os quais os dados serão baixados
* **train_time**: Intervalo de tempo no qual o treinamento será realizado
* **eval_time**: Intervalo de tempo no qual a avaliação será realizada
* **patrimony**: Valor inicial para investir
* **positions**: Lista com possíveis posições na carteira
* **max_steps**: Critério de parada pelo número máximo de iterações com ambiente
* **window_size**: Quantidade de dias anteriores em cada observação
* **epi_len**: Quantidade de passos antes que o ambiente seja reinicializado.
* **n_times**: Quantidade de vezes que experimento será executado
* **model**: Modelo que será utilizado dentro do algoritmo selecionado

Exemplo de comando para executar com RLib:

> python main.py --ticket BBAS3.SA --train_time 2000-01-01/2020-01-01 --eval_time 2020-01-01/2023-12-01 --patrimony 1000 --position 0, 1 --model FullyCon

A implementação do DQN do respositório https://github.com/udacity/deep-reinforcement-learning/tree/master, pode ser executada pelo comando:

> python .\Algorithms\DQN\train.py




## Pacotes requisitados:
* ray[rllib]
* pandas_datareader
* yfinance
