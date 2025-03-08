#7 Days of Code Machine Learning



##Dia 1/7


Vamos explorar os dados de músicas do Spotify?


Mas antes de iniciar essa exploração, 
você precisará executar um passo bem importante, 
que é a coleta e organização dos dados.

Baixe o datataset de músicas do Spotify disponível no Kaggle e realize a leitura e exploração em uma IDE da sua preferência.

Agora é hora de começar a análise exploratória dos dados, que é um processo importante para entender melhor os dados com os quais você estará trabalhando.
No caso dos dados de músicas do Spotify, ela pode nos ajudar a identificar padrões, tendências e relações entre as variáveis disponíveis.
Isso pode te fornecer insights valiosos sobre o que torna uma música popular no Spotify, 
como características musicais, duração da música, gênero musical, quais os artistas mais ouvidos e outros.


Através dessa análise inicial, você pode descobrir informações relevantes sobre os dados
e ajudar a orientar a criação de modelos de Machine Learning mais eficazes para prever a popularidade das músicas no Spotify.


Além disso, a análise exploratória pode ajudar a identificar possíveis problemas nos dados, 
como valores faltantes ou inconsistências, coisas que precisam ser tratadas antes da criação do modelo.


Então, comece investigando os seus dados e gerando visualizações para identificar padrões ou estatísticas interessantes.
Por exemplo: 
Quais são as 100 músicas mais populares?
Quais os artistas mais populares? 
Dentre os gêneros musicais, quais são mais populares?


Você pode começar gerando algumas estatísticas bem simples, 
como contar valores de alguma coluna, 
somá-los, 
fazer agrupamentos, 
visualizar a dimensão dos dados,
tipos de cada coluna, 
identificar inconsistências,
campos com valores nulos ou duplicados, etc.


Você vai perceber que essa base de dados possui características de músicas como,
duração, danceabilidade, energia, modo, popularidade, velocidade, acústico, instrumentalização, e outros. 


"Por exemplo, a danceabilidade é uma medida de quão adequada a música é para dançar, 
enquanto que a energia é uma medida de quão intensa e animada a música é."



##DICA


Você pode importar os dados diretamente do Github para seu notebook apenas passando o endereço do link “Raw” como origem.
Você pode utilizar o Jupyter Notebook, Google Colab ou outra IDE de sua preferência.
Para gerar os gráficos, existem diversas ferramentas de visualização de dados. 
Minha recomendação é que você comece usando o Matplotlib e Seaborn.
Utilize a biblioteca Pandas para extrair informações e estatísticas. 




##Dia 2/7


Vamos dar os primeiros passos em Machine Learning!


Existem três tipos principais de machine learning (aprendizado de máquina):

supervisionado, 

não supervisionado 

e por reforço. 


Neste desafio, você vai se concentrar no aprendizado supervisionado, 
que envolve a previsão de uma variável de saída com base em um conjunto de variáveis de entrada.

Indo mais a fundo, existem dois tipos de aprendizado supervisionado: classificação e regressão. 
Neste desafio, o foco será desenvolver um modelo de classificação para prever se uma música será popular ou não.

Mas para que você vai fazer isso? 
Entender se uma música será popular ou não pode ajudar a tomar melhores decisões de marketing e promover o sucesso de uma música, por exemplo.

Antes de começar a criar o seu modelos, você precisará passar pela etapa de pré-processamento de dados. 

O pré-processamento de dados é uma das etapas mais importantes no processo de Machine Learning. 

Essa é a fase em que você vai limpar, organizar e transformar os dados brutos em dados que possam ser usados para treinar os seus modelos.


Existem várias técnicas de pré-processamento de dados que você pode aplicar, como: 

remoção de dados duplicados, 

preenchimento de dados ausentes, 

normalização dos dados, 

engenharia de recursos e outros. 


Portanto, você pode começar aplicando algumas dessas técnicas aos dados disponíveis.


Outro ponto importante: 

você deve ter percebido na etapa anterior, de análise dos dados, que a coluna de popularidade apresenta números que variam de 0 a 100.

E de acordo com a documentação da API do Spotify, 100 representa a mais popular. 

Esse número é baseado no número total de reproduções que a faixa teve e quão recentes são as reproduções.


Como você quer saber se uma música será popular ou não, você precisará estabelecer um corte de popularidade.
Por exemplo, você pode definir que todas as músicas com popularidade acima de 70 são consideradas populares e aquelas abaixo de 70 são consideradas não populares. 
Esse corte não é padrão e pode ser ajustado para testar diferentes cenários.


Portanto, para prosseguir com o modelo de classificação, você precisa converter a coluna de popularidade em uma classe binária (1 para popular, 0 para não popular).



##DICA


Para realizar o processo de corte de popularidade, 
você pode criar uma nova coluna de classe utilizando o método select() da bibllioteca Numpy. 
Essa função permite criar uma nova coluna a partir de condicionais. 
É útil quando você quer aplicar diferentes regras a diferentes valores de uma coluna existente e criar uma nova coluna com os resultados.



##Dia 3/7


No desafio de hoje, você vai trabalhar na divisão dos seus dados em treino, validação e teste. 
Essa é uma etapa essencial antes de criar os seus modelos de machine learning.

Mas você deve estar se perguntando, por que eu vou precisar dividir, não é mesmo?! 
A resposta é simples: para avaliar o desempenho do seu modelo de forma justa. 
Se você usar todos os dados para treinar o modelo, não terá como saber se ele é bom o suficiente para generalizar para dados novos.
Além disso, essa técnica é usada para garantir que o modelo não esteja superajustado (overfitting) aos dados de treinamento e que possa funcionar bem em novos dados.

Em resumo, os dados de treinamento são usados para treinar o modelo, 
enquanto que os dados de teste são usados para avaliar o desempenho do modelo em dados que ele nunca viu antes. 
E os dados de validação, são usados para ajustar os hiperparâmetros do modelo (parâmetros que melhoram o desempenho do modelo).

Mas como você pode dividir os dados? Bom, existem várias formas de fazer isso.

Uma delas é a divisão aleatória, que simplesmente separa os dados em três conjuntos de forma aleatória.
Geralmente, 70-80% dos dados são usados para treino, 10-20% para teste e 10-20% para validação. 
Essa técnica é simples e rápida, mas pode não ser uma boa escolha quando há desequilíbrio de dados.

Outra forma é a validação cruzada, que é usada para avaliar a capacidade de generalização do modelo em diferentes conjuntos de dados. 
Ela ajuda muito a evitar o overfitting, que é quando um modelo se ajusta demais aos dados de treinamento, mas não generaliza bem para novos dados.

Uma forma comum de validação cruzada é a StratifiedKFold, 
que é especialmente útil para conjuntos de dados desbalanceados (e aqui já vai um spoiler do desafio do dia 6 👀).

Além disso, após a divisão dos dados, é necessário dividir o conjunto em X e Y. 
No seu caso, você terá o conjunto de variáveis explicativas (X), como gênero musical, duração da música, instrumentação, etc,
e a variável de saída (Y), que indicará a popularidade da música, e que você quer prever.

Então, minha proposta pra hoje é que você realize a divisão dos dados utilizando a validação cruzada.
E como desafio extra, utilize a StratifiedKFold e compare com outras técnicas de divisão de dados.

Lembre-se que a divisão de dados é uma etapa muito importante no processo de machine learning
e pode afetar significativamente os resultados do seu modelo.


DICA

Você pode começar separando seu dataframe em df_train e df_test aplicando o método train_test_split() da biblioteca Sklearn.

Em seguida, a partir do dataframe de treino (df_train), 
utilize a validação cruzada para separação dos dados em treino e validação.

Tente utilizar a classe StratifiedKFold e aplicar um looping para separar os dados.

