from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from iqoptionapi.stable_api import IQ_Option
import time 
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import LSTM 
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.trend import PSARIndicator
from keras.models import Sequential
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

short_window = 9
long_window = 6
model_inputs = []

# Conectando à IQ Option
I_want_money = IQ_Option("e-mail", "senha")
I_want_money.connect()

# Verificando se a conexão foi bem-sucedida
if I_want_money.check_connect():
    print('Conexão bem-sucedida!')
    # Solicita ao usuário que escolha entre a conta demo ou real
    conta_tipo = input("Por favor, escolha o tipo de conta (demo/real): ").strip().lower()

    if conta_tipo == 'demo':
        I_want_money.change_balance('PRACTICE')  # Muda para a conta demo
    elif conta_tipo == 'real':
        I_want_money.change_balance('REAL')  # Muda para a conta real
    else:
        print("Entrada inválida. Por favor, reinicie e escolha 'demo' ou 'real'.")
        exit()  # Encerra o script se a entrada for inválida
else:
    print('Erro na conexão.')
    I_want_money.connect()

# Definição das funções de compra e venda
def comprar(pairs, valor, tempo):
    I_want_money.buy(valor, pair, "call", tempo)

def vender(pairs, valor, tempo):
    I_want_money.buy(valor, pair, "put", tempo)     

# Tentativa de carregar modelo anterior
try:
    model_1 = load_model("meu_modelo_1.keras")
except:
    model_1 = None


# Loop de aprendizado contínuo
for day in range(1):
    print(f"Iniciando analise dos dados")

    # Coletando dados dos pares de moedas para vários timeframes
end_from_time = time.time()
two_years_in_seconds = 60 * 60 * 24 * 365 * 12
pairs_digitais_binarias = ["EURUSD"]


 # Combine todas as listas em uma lista total de pairs
all_pairs = pairs_digitais_binarias 

timeframes_seconds = [60, 300, 3600]
all_data = []

for pair in all_pairs:  
    for timeframe in timeframes_seconds:
        print(f"solicitando dados para {pair} no timeframe de {timeframe} segundos")
        data = I_want_money.get_candles(pair, timeframe, int((end_from_time - two_years_in_seconds) / timeframe), end_from_time)
        df_temp = pd.DataFrame(data)
        df_temp['timeframe'] = timeframe  # Adiciona uma coluna para identificar o timeframe
        df_temp['pairs'] = pair  # Adiciona uma coluna para identificar o pair
        all_data.append(df_temp)
        time.sleep(5)
        
df = pd.concat(all_data, ignore_index=True)
print(df.head())

# Adicione a coluna 'retorno' aqui
df['retorno'] = df['close'].pct_change()

     # Convertendo a coluna de tempo para o formato datetime
df['time'] = pd.to_datetime(df['from'], unit='s')
    
     # Implementação de características adicionais
df['media_movel_20'] = df['close'].rolling(window=20).mean()
df['banda_bollinger_superior'], df['banda_bollinger_inferior'] = df['close'].rolling(window=20).mean() + 2*df['close'].rolling(window=20).std(), df['close'].rolling(window=20).mean() - 2*df['close'].rolling(window=20).std()
df['banda_bollinger_superior'], df['banda_bollinger_inferior'] = df['close'].rolling(window=20).mean() + 2*df['close'].rolling(window=20).std(), df['close'].rolling(window=20).mean() - 2*df['close'].rolling(window=20).std()
    
     # Identificação de divergências para as Bandas de Bollinger
df['bollinger_divergencia'] = 0
df.loc[(df['close'] > df['close'].shift(1)) & (df['close'] < df['banda_bollinger_inferior'].shift(1)), 'bollinger_divergencia'] = 1  # Divergência positiva
df.loc[(df['close'] < df['close'].shift(1)) & (df['close'] > df['banda_bollinger_superior'].shift(1)), 'bollinger_divergencia'] = -1  # Divergência negativa
    
bollinger_divergencia_positiva = df[df['bollinger_divergencia'] == 1]
bollinger_divergencia_negativa = df[df['bollinger_divergencia'] == -1]

if not bollinger_divergencia_positiva.empty:
    print("Divergência positiva de Bollinger identificada nas seguintes datas:")
    print(bollinger_divergencia_positiva.index)

if not bollinger_divergencia_negativa.empty:
    print("\nDivergência negativa de Bollinger identificada nas seguintes datas:")
    print(bollinger_divergencia_negativa.index)
        
     # Cálculo do RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).fillna(0)
loss = (-delta.where(delta < 0, 0)).fillna(0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))

     # Identificação de divergências RSI
df['rsi_divergencia'] = 0  # Cria uma nova coluna com valores padrão 0
df.loc[(df['close'] > df['close'].shift(1)) & (df['rsi'] < df['rsi'].shift(1)), 'rsi_divergencia'] = 1  # Divergência positiva
df.loc[(df['close'] < df['close'].shift(1)) & (df['rsi'] > df['rsi'].shift(1)), 'rsi_divergencia'] = -1  # Divergência negativa

rsi_divergencia_positiva = df[df['rsi_divergencia'] == 1]
rsi_divergencia_negativa = df[df['rsi_divergencia'] == -1]

if not rsi_divergencia_positiva.empty:
    print("Divergência positiva de RSI identificada nas seguintes datas:")
    print(rsi_divergencia_positiva.index)

if not rsi_divergencia_negativa.empty:
    print("\nDivergência negativa de RSI identificada nas seguintes datas:")
    print(rsi_divergencia_negativa.index)
        
     # Implementação do indicador MACD
macd = MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_diff'] = macd.macd_diff()
    
     # Identificação de divergências MACD
df['macd_divergencia'] = 0  # Cria uma nova coluna com valores padrão 0
df.loc[(df['close'] > df['close'].shift(1)) & (df['macd'] < df['macd'].shift(1)), 'macd_divergencia'] = 1  # Divergência positiva
df.loc[(df['close'] < df['close'].shift(1)) & (df['macd'] > df['macd'].shift(1)), 'macd_divergencia'] = -1  # Divergência negativa

macd_divergencia_positiva = df[df['macd_divergencia'] == 1]
macd_divergencia_negativa = df[df['macd_divergencia'] == -1]

if not macd_divergencia_positiva.empty:
    print("Divergência positiva de MACD identificada nas seguintes datas:")
    print(macd_divergencia_positiva.index)

if not macd_divergencia_negativa.empty:
    print("\nDivergência negativa de MACD identificada nas seguintes datas:")
    print(macd_divergencia_negativa.index)
        
 
     # Cálculo de Médias Móveis
df['media_movel_curta'] = df['close'].rolling(window=5).mean()
df['media_movel_longa'] = df['close'].rolling(window=20).mean()
    
     # Identificação de Convergências
df['convergencia'] = 0  # Cria uma nova coluna com valores padrão 0
df.loc[df['media_movel_curta'] > df['media_movel_longa'], 'convergencia'] = 1  # Marca pontos de convergência positiva com 1
df.loc[df['media_movel_curta'] < df['media_movel_longa'], 'convergencia'] = -1  # Marca pontos de convergência negativa com -1

       # Verificando e imprimindo pontos de convergência
convergencias_previas = df['convergencia'].shift(1)
for index, (convergencia_atual, convergencia_anterior) in enumerate(zip(df['convergencia'], convergencias_previas)):
    if pd.notnull(convergencia_anterior):  # Ignora a primeira linha onde não há valor anterior
        if convergencia_atual == 1 and convergencia_anterior != 1:
            print(f"Convergência positiva no índice {index}")
        elif convergencia_atual == -1 and convergencia_anterior != -1:
            print(f"Convergência negativa no índice {index}")



# Pin Bar
df['pin_bar'] = np.where(
    ((df['open'] - df['min'] > 2 * (df['close'] - df['open'])) & (df['close'] > df['open'])) |
    ((df['max'] - df['close'] > 2 * (df['open'] - df['close'])) & (df['close'] < df['open'])),
    1, 0)


 #suporte e resistencia
df['resistance'] = df['max'].shift(1)
df['support'] = df['min'].shift(1)

 #medias moveis
df['trend'] = np.where(df['media_movel_curta'] > df['media_movel_longa'], 1, -1)  # 1 para alta, -1 para baixa


# Bullish Engulfing
df['bullish_engulfing'] = np.where((df['close'].shift(1) < df['open'].shift(1)) &
                                   (df['open'] < df['close'].shift(1)) &
                                   (df['close'] > df['open'].shift(1)), 1, 0)

# Bearish Engulfing
df['bearish_engulfing'] = np.where((df['close'].shift(1) > df['open'].shift(1)) &
                                   (df['open'] > df['close'].shift(1)) &
                                   (df['close'] < df['open'].shift(1)), 1, 0)

# Doji
df['doji'] = np.where(abs(df['open'] - df['close']) <= (df['max'] - df['min']) * 0.1, 1, 0)

# Bullish Harami
df['bullish_harami'] = np.where((df['close'].shift(1) < df['open'].shift(1)) &
                                (df['open'] > df['close'].shift(1)) &
                                (df['close'] < df['open'].shift(1)), 1, 0)

# Bearish Harami
df['bearish_harami'] = np.where((df['close'].shift(1) > df['open'].shift(1)) &
                                (df['open'] < df['close'].shift(1)) &
                                (df['close'] > df['open'].shift(1)), 1, 0)

# Morning Star
df['morning_star'] = np.where((df['close'].shift(2) < df['open'].shift(2)) &
                              (df['close'].shift(1) < df['open'].shift(1)) &
                              (df['open'] > df['close'].shift(1)) &
                              (df['close'] > df['open']), 1, 0)

# Evening Star
df['evening_star'] = np.where((df['close'].shift(2) > df['open'].shift(2)) &
                              (df['close'].shift(1) > df['open'].shift(1)) &
                              (df['open'] < df['close'].shift(1)) &
                              (df['close'] < df['open']), 1, 0)

# Three Black Crows
df['three_black_crows'] = np.where((df['close'].shift(2) > df['open'].shift(2)) & 
                                   (df['close'].shift(1) > df['open'].shift(1)) & 
                                   (df['close'] > df['open']) & 
                                   (df['close'].shift(2) > df['close'].shift(1)) &
                                   (df['close'].shift(1) > df['close']), 1, 0)


 #Three White Soldiers

df['three_white_soldiers'] = np.where((df['open'].shift(2) > df['close'].shift(2)) &
(df['open'].shift(1) > df['close'].shift(1)) &
(df['open'] > df['close']) &
(df['close'].shift(2) < df['close'].shift(1)) &
(df['close'].shift(1) < df['close']), 1, 0)


# Piercing Line
df['piercing_line'] = np.where((df['close'].shift(1) < df['open'].shift(1)) & 
                               (df['open'] < df['close'].shift(1)) & 
                               (df['close'] > df['close'].shift(1) + 0.5*(df['open'].shift(1) - df['close'].shift(1))), 1, 0)

#OCO 
df['OCO'] = np.where((df['close'].shift(2) < df['close'].shift(1)) & (df['close'].shift(1) > df['close']) & (df['close'].shift(2) < df['close']), 1, 0)


#topo duplo
df['topo_duplo'] = np.where((df['close'].shift(2) > df['close'].shift(1)) & (df['close'].shift(1) < df['close']) & (df['close'].shift(2) < df['close']), 1, 0)


#fundo duplo 
df['fundo_duplo'] = np.where((df['close'].shift(2) < df['close'].shift(1)) & (df['close'].shift(1) > df['close']) & (df['close'].shift(2) > df['close']), 1, 0)


#BEARISH PIVOT 
df['bearish_pivot'] = np.where((df['close'].shift(2) < df['close'].shift(1)) & (df['close'].shift(1) > df['close']), 1, 0)


#BULLISH PIVOT
df['bullish_pivot'] = np.where((df['close'].shift(2) > df['close'].shift(1)) & (df['close'].shift(1) < df['close']), 1, 0)


#sem mechas 
df['bullish_abandoned_baby'] = np.where((df['open'].shift(2) > df['close'].shift(2)) & (df['open'].shift(1) == df['close'].shift(1)) & (df['open'] < df['close']), 1, 0)


#Abandoned baby(sem mechas )
df['bearish_abandoned_baby'] = np.where((df['open'].shift(2) < df['close'].shift(2)) & (df['open'].shift(1) == df['close'].shift(1)) & (df['open'] > df['close']), 1, 0)


#three line 
df['three_line_strike_bullish'] = np.where((df['close'].shift(3) > df['open'].shift(3)) & 
                                          (df['close'].shift(2) > df['open'].shift(2)) & 
                                          (df['close'].shift(1) > df['open'].shift(1)) & 
                                          (df['open'] < df['close'].shift(1)) & 
                                          (df['close'] < df['open'].shift(3)), 1, 0)


#three line strike
df['three_line_strike_bearish'] = np.where((df['close'].shift(3) < df['open'].shift(3)) & 
                                          (df['close'].shift(2) < df['open'].shift(2)) & 
                                          (df['close'].shift(1) < df['open'].shift(1)) & 
                                          (df['open'] > df['close'].shift(1)) & 
                                          (df['close'] > df['open'].shift(3)), 1, 0)


#two black
df['two_black_gapping'] = np.where((df['close'].shift(2) > df['open'].shift(2)) & 
                                   (df['open'].shift(2) > df['close'].shift(1)) & 
                                   (df['close'].shift(1) > df['open'].shift(1)) & 
                                   (df['close'] > df['open']), 1, 0)

#three stars south
df['three_stars_in_south'] = np.where((df['close'].shift(2) > df['open'].shift(2)) & 
                                     (df['close'].shift(1) < df['close'].shift(2)) & 
                                     (df['close'] < df['close'].shift(1)), 1, 0)


#stick sandwich
df['stick_sandwich'] = np.where((df['close'].shift(2) > df['open'].shift(2)) & 
                                (df['close'].shift(1) < df['open'].shift(1)) & 
                                (df['close'] > df['open']) & 
                                (df['close'] == df['close'].shift(2)), 1, 0)


# Bandeira/flâmula após um movimento de alta
df['bandeira_alta'] = np.where((df['close'].shift(2) > df['media_movel_curta'].shift(2)) & 
                               (df['close'].shift(1) < df['media_movel_curta'].shift(1)) & 
                               (df['close'] > df['media_movel_curta']), 1, 0)

# Bandeira/flâmula após um movimento de baixa
df['bandeira_baixa'] = np.where((df['close'].shift(2) < df['media_movel_curta'].shift(2)) & 
                                (df['close'].shift(1) > df['media_movel_curta'].shift(1)) & 
                                (df['close'] < df['media_movel_curta']), 1, 0)


# Triângulo de Alta
df['triangulo_alta'] = np.where((df['media_movel_curta'] > df['media_movel_longa']) & 
                                (df['media_movel_curta'].shift(1) < df['media_movel_longa'].shift(1)), 1, 0)

# Triângulo de Baixa
df['triangulo_baixa'] = np.where((df['media_movel_curta'] < df['media_movel_longa']) & 
                                 (df['media_movel_curta'].shift(1) > df['media_movel_longa'].shift(1)), 1, 0)



#media movel50
df['media_movel_50'] = df['close'].rolling(window=50).mean()


#desvio padrão dos retornos
df['volatilidade'] = df['retorno'].rolling(window=50).std()

#modelo de regressão 
modelo = LinearRegression().fit(df[['close', 'open']], df['close'])

#teste de estacionaridade 
resultado = adfuller(df['close'])

#modelos Arima 
modelo = ARIMA(df['close'], order=(5,1,0))
modelo_fit = modelo.fit()


#retorno histórico 
df['retorno'] = df['close'].pct_change()


# Adicionando recurso de momento
df['momentum'] = df['close'] - df['close'].shift(4)

# Decompondo o timestamp
df['datetime'] = pd.to_datetime(df['time'], unit='s')  # Convertendo timestamp para datetime
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek

# Adicionando recurso de volatilidade
df['daily_returns'] = df['close'].pct_change()
  
# Removendo linhas com valores NaN
df = df.dropna()

# Visualizando os novos recursos
print(df.head())

# Coluna 'target'
df['target'] = df['close'].diff().apply(lambda x: 1 if x > 0 else 0)


# Especificando as colunas dos indicadores no DataFrame para serem usadas como features
print(df.columns)
X = df[[
    'close', 'open', 'max', 'min', 'media_movel_20', 'banda_bollinger_superior',
    'banda_bollinger_inferior', 'bollinger_divergencia', 'media_movel_curta',
    'media_movel_longa', 'convergencia', 'macd', 'macd_signal', 'macd_divergencia',
    'rsi', 'rsi_divergencia', 'pin_bar', 'resistance', 'support', 'trend',
    'bullish_engulfing', 'bearish_engulfing', 'doji', 'bullish_harami',
    'bearish_harami', 'morning_star', 'evening_star', 'three_black_crows', 'three_white_soldiers',
    'bullish_abandoned_baby', 'bearish_abandoned_baby', 'piercing_line',
    'topo_duplo', 'fundo_duplo', 'bearish_pivot', 'bullish_pivot', 'OCO',
    'three_line_strike_bullish', 'three_line_strike_bearish', 'two_black_gapping',
    'three_stars_in_south', 'stick_sandwich', 'bandeira_alta', 'bandeira_baixa',
    'triangulo_alta', 'triangulo_baixa', 'retorno', 'media_movel_50', 'volatilidade']]

missing_columns = [col for col in X if col not in df.columns]
if missing_columns:
    print("Colunas faltando:", missing_columns)
else:
    print("Todas as colunas estão presentes.")
    
y = df['target']


       # Dividindo os dados para treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalizando os dados
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

         # Remodelando os dados para LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], 1)
print(X_train.shape)
print(X_test.shape)
    
     # Definindo a Primeira Rede LSTM (RN1)
model_1 = Sequential()
model_1.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(49, 1)))
model_1.add(Dropout(0.2))
model_1.add(LSTM(25, activation='tanh', return_sequences=True))  # Camada LSTM extra
model_1.add(Dropout(0.2))
model_1.add(Dense(1, activation='sigmoid'))

optimizer_1 = RMSprop(learning_rate=0.003)
model_1.compile(optimizer=optimizer_1, loss='binary_crossentropy', metrics=['accuracy'])
    
     # Implementação de callbacks
lr_adjuster_1 = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=6, verbose=3, min_lr=0.00003)
early_stopper_1 = EarlyStopping(patience=5)


 # Treinando o Model_1
print("Começando o treinamento do Model_1.")
model_1.fit(X_train, y_train, epochs=2500, batch_size=128)  
print("Treinamento do Model_1 concluído.")

     # Salvar o Model_1
model_1.save("meu_modelo_1.keras")
print(f"Modelo 1 atualizado e salvo.")


while True:
  
    
    indices_de_erros = []
    
        # Previsões do modelo
    predictions_1 = model_1.predict(X_test)

    start_index = 0
    
    for i in range(start_index, len(df)):
         
         #condição de compra
        condicao_compra_1 = X['media_movel_curta'].iloc[i] > X['media_movel_longa'].iloc[i]
        condicao_compra_2 = X['rsi'].iloc[i] < 30
        condicao_compra_3 = X['macd'].iloc[i] > X['macd_signal'].iloc[i]
        condicao_compra_4 = X['bollinger_divergencia'].iloc[i] == 1
        condicao_compra_5 = X['close'].iloc[i] < X['banda_bollinger_inferior'].iloc[i]  
        condicao_compra_6 = X['convergencia'].iloc[i] == 1  
        
        print(f"Condições de compra: {[condicao_compra_1, condicao_compra_2, condicao_compra_3, condicao_compra_4, condicao_compra_5, condicao_compra_6]}")
        
        
         # Condições de Venda
        condicao_venda_1 = X['media_movel_curta'].iloc[i] < X['media_movel_longa'].iloc[i]
        condicao_venda_2 = X['rsi'].iloc[i] > 70
        condicao_venda_3 = X['macd'].iloc[i] < X['macd_signal'].iloc[i]
        condicao_venda_4 = X['bollinger_divergencia'].iloc[i] == -1
        condicao_venda_5 = X['close'].iloc[i] > X['banda_bollinger_superior'].iloc[i]  
        condicao_venda_6 = X['convergencia'].iloc[i] == -1  

        print(f"Condições de venda: {[condicao_venda_1, condicao_venda_2, condicao_venda_3, condicao_venda_4, condicao_venda_5, condicao_venda_6]}")
        
           # Verifica se duas ou mais condições de compra são verdadeiras
        if sum([condicao_compra_1, condicao_compra_2, condicao_compra_3, condicao_compra_4, condicao_compra_5,  condicao_compra_6]) >= 4:
            print("Tomando decisão de compra")
            comprar('pairs', 100, 1)  
         # Verifica se duas ou mais condições de venda são verdadeiras
        elif sum([condicao_venda_1, condicao_venda_2, condicao_venda_3, condicao_venda_4, condicao_venda_5, condicao_venda_6]) >= 4:
            print("Tomando decisão de venda")
            vender('pairs', 100, 1)  
        
          # Adicione um atraso entre os trades para evitar a execução demasiado rápida de ordens
        time.sleep(600)
            # Adicionando os casos de erro ao conjunto de dados de treinamento
        X_train = np.concatenate((X_train, X_test[indices_de_erros]))
        y_train = np.concatenate((y_train, y_test.iloc[indices_de_erros]))  
         
           # Salvando o DataFrame df em um arquivo CSV para futuros treinamentos
    df.to_csv('analises_e_entradas.csv', index=False)
    print("Dados salvos com sucesso no arquivo 'analises_e_entradas.csv'")
    
   # Salvando o modelo atualizado
    model_1.save("meu_modelo_1.keras")
    print(f"Modelos atualizados e salvo no dia")
        
    time.sleep(60)