#-*- coding: utf-8 -*-
##------------------------------------------------------------------------
## Case: Cobrança
## Autor: Prof. Roberto Ãngelo
##------------------------------------------------------------------------

# Bibliotecas padrão
import DM_Utils as du
import numpy as np
import pandas as pd
        
## Carregando os dados

print()
print('--> INÍCIO DE CARGA DE DADOS')

## Carregando os dados
dataset = pd.read_csv('BASE.txt',sep='\t') # Separador TAB

print('--> FIM DA CARGA DE DADOS')

#print(dataset.describe())
#------------------------------------------------------------------------------------------
# Pré-processamento das variáveis de entrada
#------------------------------------------------------------------------------------------
# ATRIBUTO CD_SETOR LOJA
dataset['pre_CD_SETOR_11_8'] = [1 if x in (11,8) else 0 for x in dataset['CD_SETOR_LOJA']]
dataset['pre_CD_SETOR_1_12_14'] = [1 if x in (1,12,14) else 0 for x in dataset['CD_SETOR_LOJA']]
dataset['pre_CD_SETOR_4'] = [1 if x==4 else 0 for x in dataset['CD_SETOR_LOJA']]
dataset['pre_CD_SETOR_6'] = [1 if x==6 else 0 for x in dataset['CD_SETOR_LOJA']]
dataset['pre_CD_SETOR_9_10'] = [1 if x in (9,10) else 0 for x in dataset['CD_SETOR_LOJA']]
dataset['pre_CD_SETOR_13_5_3_7'] = [1 if x in (13,5,3,7) else 0 for x in dataset['CD_SETOR_LOJA']]
# ATRIBUTO VALOR
dataset['pre_VALOR'] = [200 if np.isnan(x) or x < 200 else x for x in dataset['VALOR']] 
dataset['pre_VALOR'] = [1700 if x > 1700 else x for x in dataset['pre_VALOR']] 
dataset['pre_VALOR'] = [(x-200)/(1700-200) for x in dataset['pre_VALOR']] 
# ATRIBUTO DIAS_PRIMEIRA_PARCELA
dataset['pre_PRI_PARCELA'] = [0 if np.isnan(x) or x < 0 else x for x in dataset['DIAS_PRIMEIRA_PARCELA']] 
dataset['pre_PRI_PARCELA'] = [70 if x > 70 else x for x in dataset['pre_PRI_PARCELA']] 
dataset['pre_PRI_PARCELA'] = [x/70 for x in dataset['pre_PRI_PARCELA']] 
# ATRIBUTO RELACIONAMENTO
dataset['pre_RELAC_OUTRO_DESC'] = [1 if x in ('OUTRO','DESCONHECIDO')  else 0 for x in dataset['RELACIONAMENTO']]
dataset['pre_RELAC_VIP'] = [1 if x=='VIP' else 0 for x in dataset['RELACIONAMENTO']]
dataset['pre_RELAC_CONHECIDO'] = [1 if x== 'CONHECIDO' else 0 for x in dataset['RELACIONAMENTO']]
dataset['pre_RELAC_POUCO_RELAC'] = [1 if x== 'POUCO RELACIONAMENTO' else 0 for x in dataset['RELACIONAMENTO']]
dataset['pre_RELAC_REPESCAGEM'] = [1 if x== 'REPESCAGEM' else 0 for x in dataset['RELACIONAMENTO']]
# ATRIBUTO TIPO_CLIENTE
dataset['pre_TIPO_CLIENTE_A'] = [1 if x=='A'  else 0 for x in dataset['TIPO_CLIENTE']]
dataset['pre_TIPO_CLIENTE_B'] = [1 if x=='B' else 0 for x in dataset['TIPO_CLIENTE']]
dataset['pre_TIPO_CLIENTE_C'] = [1 if x=='C' else 0 for x in dataset['TIPO_CLIENTE']]
# ATRIBUTO NUMERO_PARCELA
dataset['pre_NUMERO_PARCELA'] = [0 if np.isnan(x) or x < 0 else x for x in dataset['NUMERO_PARCELA']] 
dataset['pre_NUMERO_PARCELA'] = [6 if x > 6 else x for x in dataset['pre_NUMERO_PARCELA']] 
dataset['pre_NUMERO_PARCELA'] = [x/6 for x in dataset['pre_NUMERO_PARCELA']] 
# ATRIBUTO CD_SEXO
dataset['pre_CD_SEXO_M'] = [1 if x=='M'  else 0 for x in dataset['CD_SEXO']]
dataset['pre_CD_SEXO_F'] = [1 if x=='F' else 0 for x in dataset['CD_SEXO']]
dataset['pre_CD_SEXO_I'] = [1 if x=='I' else 0 for x in dataset['CD_SEXO']]
# ATRIBUTO CEP_3
dataset['pre_CEP_3_0'] = [0 if np.isnan(x) or x < 0 else 0 for x in dataset['CEP_3']] 
dataset['pre_CEP_3_600-699'] = [1 if x >= 600 or x <= 699 else 0 for x in dataset['CEP_3']]
dataset['pre_CEP_3_300-399'] = [1 if x >= 300 or x <= 399 else 0 for x in dataset['CEP_3']]
dataset['pre_CEP_3_700-799'] = [1 if x >= 700 or x <= 799 else 0 for x in dataset['CEP_3']]
dataset['pre_CEP_3_800-899'] = [1 if x >= 800 or x <= 899 else 0 for x in dataset['CEP_3']]
dataset['pre_CEP_3_900-999'] = [1 if x >= 900 or x <= 999 else 0 for x in dataset['CEP_3']]
dataset['pre_CEP_3_500-599'] = [1 if x >= 500 or x <= 599 else 0 for x in dataset['CEP_3']]
dataset['pre_CEP_3_100-199'] = [1 if x >= 100 or x <= 199 else 0 for x in dataset['CEP_3']]
dataset['pre_CEP_3_0-99'] = [1 if x >= 0 or x <= 99 else 0 for x in dataset['CEP_3']]
dataset['pre_CEP_3_200-299'] = [1 if x >= 200 or x <= 299 else 0 for x in dataset['CEP_3']]
dataset['pre_CEP_3_400-499'] = [1 if x >= 400 or x <= 499 else 0 for x in dataset['CEP_3']]
# ATRIBUTO FLAG_ENDERECO_VALIDO
dataset['pre_FLAG_ENDERECO_VALIDO_S'] = [1 if x=='Sim' else 0 for x in dataset['FLAG_ENDERECO_VALIDO']]
dataset['pre_FLAG_ENDERECO_VALIDO_N'] = [1 if x=='Não' else 0 for x in dataset['FLAG_ENDERECO_VALIDO']]
# ATRIBUTO QTD_OUTRAS_DIVIDAS
dataset['pre_QTD_OUTRAS_DIVIDAS_0-3'] = [1 if x >= 0 or x <= 3 else 0 for x in dataset['QTD_OUTRAS_DIVIDAS']]
dataset['pre_QTD_OUTRAS_DIVIDAS_4-7'] = [1 if x >= 4 or x <= 7 else 0 for x in dataset['QTD_OUTRAS_DIVIDAS']]
dataset['pre_QTD_OUTRAS_DIVIDAS_8-11'] = [1 if x >= 8 or x <= 11 else 0 for x in dataset['QTD_OUTRAS_DIVIDAS']]
dataset['pre_QTD_OUTRAS_DIVIDAS_12-16'] = [1 if x >= 12 or x <= 16 else 0 for x in dataset['QTD_OUTRAS_DIVIDAS']]
# ATRIBUTO QTD_EMPRESTIMO_NEG_ANT
dataset['pre_QTD_EMPRESTIMO_NEG_ANT_1'] = [1 if x == 1 else 0 for x in dataset['QTD_EMPRESTIMO_NEG_ANT']]
dataset['pre_QTD_EMPRESTIMO_NEG_ANT_2'] = [1 if x == 2 else 0 for x in dataset['QTD_EMPRESTIMO_NEG_ANT']]
dataset['pre_QTD_EMPRESTIMO_NEG_ANT_3'] = [1 if x == 3 else 0 for x in dataset['QTD_EMPRESTIMO_NEG_ANT']]
dataset['pre_QTD_EMPRESTIMO_NEG_ANT_6+'] = [1 if x >= 6 else 0 for x in dataset['QTD_EMPRESTIMO_NEG_ANT']]
# ATRIBUTO RISCO_CLIENTE
dataset['pre_RISCO_CLIENTE_B'] = [1 if x == 'B' else 0 for x in dataset['RISCO_CLIENTE']]
dataset['pre_RISCO_CLIENTE_A'] = [1 if x == 'A' else 0 for x in dataset['RISCO_CLIENTE']]
dataset['pre_RISCO_CLIENTE_C'] = [1 if x == 'C' else 0 for x in dataset['RISCO_CLIENTE']]
dataset['pre_RISCO_CLIENTE_D'] = [1 if x == 'D' else 0 for x in dataset['RISCO_CLIENTE']]
dataset['pre_RISCO_CLIENTE_E'] = [1 if x == 'E' else 0 for x in dataset['RISCO_CLIENTE']]
dataset['pre_RISCO_CLIENTE_F'] = [1 if x == 'F' else 0 for x in dataset['RISCO_CLIENTE']]
dataset['pre_RISCO_CLIENTE_G'] = [1 if x == 'G' else 0 for x in dataset['RISCO_CLIENTE']]
# ATRIBUTO IDADE
dataset['pre_IDADE'] = [18 if np.isnan(x) or x < 18 else x for x in dataset['IDADE']] 
dataset['pre_IDADE'] = [75 if x > 75 else x for x in dataset['pre_IDADE']] 
dataset['pre_IDADE'] = [(x-18)/(75-18) for x in dataset['pre_IDADE']] 
# ATRIBUTO TIPO_VINCULO
dataset['pre_TIPO_VINCULO_COOP'] = [1 if x == 'COOPERATIVO' else 0 for x in dataset['TIPO_VINCULO']]
dataset['pre_TIPO_VINCULO_OUTRO'] = [1 if x == 'OUTRO' else 0 for x in dataset['TIPO_VINCULO']]
dataset['pre_TIPO_VINCULO_PRIV'] = [1 if x == 'PRIVADO' else 0 for x in dataset['TIPO_VINCULO']]
dataset['pre_TIPO_VINCULO_PUB'] = [1 if x == 'PUBLICO' else 0 for x in dataset['TIPO_VINCULO']]
# ATRIBUTO TEMPO_DE_CLIENTE
#dataset['pre_TEMPO_DE_CLIENTE_0'] = [0 if str(x) else 0 for x in dataset['TEMPO_DE_CLIENTE']]
dataset['pre_TEMPO_DE_CLIENTE_1'] = [1 if str(x) == 'ATÉ 1 ANO' else 0 for x in dataset['TEMPO_DE_CLIENTE']]
dataset['pre_TEMPO_DE_CLIENTE_2'] = [1 if str(x) == 'ENTRE 1 E 2 ANOS' else 0 for x in dataset['TEMPO_DE_CLIENTE']]
dataset['pre_TEMPO_DE_CLIENTE_3'] = [1 if str(x) == 'ENTRE 2 E 4 ANOS' else 0 for x in dataset['TEMPO_DE_CLIENTE']]
dataset['pre_TEMPO_DE_CLIENTE_4'] = [1 if str(x) == 'ENTRE 4 E 9 ANOS' else 0 for x in dataset['TEMPO_DE_CLIENTE']]
dataset['pre_TEMPO_DE_CLIENTE_5'] = [1 if str(x) == 'ENTRE 9 E 14 ANOS' else 0 for x in dataset['TEMPO_DE_CLIENTE']]
dataset['pre_TEMPO_DE_CLIENTE_6'] = [1 if str(x) == 'MAIOR OU IGUAL A 15' else 0 for x in dataset['TEMPO_DE_CLIENTE']]
# ATRIBUTO RAMO
dataset['pre_RAMO_ALIM'] = [1 if x == 'ALIMENTOS' else 0 for x in dataset['RAMO']]
dataset['pre_RAMO_ATAC'] = [1 if x == 'ATACADO DE JOIA' else 0 for x in dataset['RAMO']]
dataset['pre_RAMO_AUTO'] = [1 if x == 'AUTO' else 0 for x in dataset['RAMO']]
dataset['pre_RAMO_CONF'] = [1 if x == 'CONFECCOES DE ATACADO' else 0 for x in dataset['RAMO']]
dataset['pre_RAMO_COSM'] = [1 if x == 'COSMETICOS' else 0 for x in dataset['RAMO']]
dataset['pre_RAMO_ELET'] = [1 if x == 'ELETRODOMESTICOS' else 0 for x in dataset['RAMO']]
dataset['pre_RAMO_MAT'] = [1 if x == 'MATERIAIS DE CONSTRUCAO' else 0 for x in dataset['RAMO']]
dataset['pre_RAMO_MOT'] = [1 if x == 'MOTOS' else 0 for x in dataset['RAMO']]
dataset['pre_RAMO_MOV'] = [1 if x == 'MOVEIS' else 0 for x in dataset['RAMO']]
dataset['pre_RAMO_OUT'] = [1 if x == 'OUTROS' else 0 for x in dataset['RAMO']]
dataset['pre_RAMO_POST'] = [1 if x == 'POSTO DE GASOLINA' else 0 for x in dataset['RAMO']]
dataset['pre_RAMO_ROUP'] = [1 if x == 'ROUPAS E CALÇADOS' else 0 for x in dataset['RAMO']]
# QTD_PARCELAS_EMPRESTIMO
dataset['pre_QTD_PARCELAS_EMPRESTIMO_1'] = [1 if x == 1 else 0 for x in dataset['QTD_PARCELAS_EMPRESTIMO']]
dataset['pre_QTD_PARCELAS_EMPRESTIMO_2-6'] = [1 if x in (2,3,4,6) else 0 for x in dataset['QTD_PARCELAS_EMPRESTIMO']]
dataset['pre_QTD_PARCELAS_EMPRESTIMO_5'] = [1 if x == 5 else 0 for x in dataset['QTD_PARCELAS_EMPRESTIMO']]
## Verifica se as variáveis numéricas foram criadas corretamente
verifica_variaveis = dataset.describe()

# ---------------------------------------------------------------------------
# Preparando os cojuntos
# ---------------------------------------------------------------------------
cols_in =  [
'pre_CD_SETOR_11_8'
,'pre_CD_SETOR_1_12_14'
,'pre_CD_SETOR_4'
,'pre_CD_SETOR_6'
,'pre_CD_SETOR_9_10'
,'pre_CD_SETOR_13_5_3_7'
,'pre_VALOR'
,'pre_PRI_PARCELA'
,'pre_RELAC_OUTRO_DESC'
,'pre_RELAC_VIP'
,'pre_RELAC_CONHECIDO'
,'pre_RELAC_POUCO_RELAC'
,'pre_RELAC_REPESCAGEM'
,'pre_TIPO_CLIENTE_A'
,'pre_TIPO_CLIENTE_B'
,'pre_TIPO_CLIENTE_C'
,'pre_NUMERO_PARCELA'
,'pre_CD_SEXO_M'
,'pre_CD_SEXO_F'
,'pre_CD_SEXO_I'
,'pre_CEP_3_0'
,'pre_CEP_3_600-699'
,'pre_CEP_3_300-399'
,'pre_CEP_3_700-799'
,'pre_CEP_3_800-899'
,'pre_CEP_3_900-999'
,'pre_CEP_3_500-599'
,'pre_CEP_3_100-199'
,'pre_CEP_3_0-99'
,'pre_CEP_3_200-299'
,'pre_CEP_3_400-499'
,'pre_FLAG_ENDERECO_VALIDO_S'
,'pre_FLAG_ENDERECO_VALIDO_N'
,'pre_QTD_OUTRAS_DIVIDAS_0-3'
,'pre_QTD_OUTRAS_DIVIDAS_4-7'
,'pre_QTD_OUTRAS_DIVIDAS_8-11'
,'pre_QTD_OUTRAS_DIVIDAS_12-16'
,'pre_QTD_EMPRESTIMO_NEG_ANT_1'
,'pre_QTD_EMPRESTIMO_NEG_ANT_2'
,'pre_QTD_EMPRESTIMO_NEG_ANT_3'
,'pre_QTD_EMPRESTIMO_NEG_ANT_6+'
,'pre_RISCO_CLIENTE_B'
,'pre_RISCO_CLIENTE_A'
,'pre_RISCO_CLIENTE_C'
,'pre_RISCO_CLIENTE_D'
,'pre_RISCO_CLIENTE_E'
,'pre_RISCO_CLIENTE_F'
,'pre_RISCO_CLIENTE_G'
,'pre_IDADE'
,'pre_IDADE'
,'pre_IDADE'
,'pre_TIPO_VINCULO_COOP'
,'pre_TIPO_VINCULO_OUTRO'
,'pre_TIPO_VINCULO_PRIV'
,'pre_TIPO_VINCULO_PUB'
,'pre_TEMPO_DE_CLIENTE_1'
,'pre_TEMPO_DE_CLIENTE_2'
,'pre_TEMPO_DE_CLIENTE_3'
,'pre_TEMPO_DE_CLIENTE_4'
,'pre_TEMPO_DE_CLIENTE_5'
,'pre_TEMPO_DE_CLIENTE_6'
,'pre_RAMO_ALIM'
,'pre_RAMO_ATAC'
,'pre_RAMO_AUTO'
,'pre_RAMO_CONF'
,'pre_RAMO_COSM'
,'pre_RAMO_ELET'
,'pre_RAMO_MAT'
,'pre_RAMO_MOT'
,'pre_RAMO_MOV'
,'pre_RAMO_OUT'
,'pre_RAMO_POST'
,'pre_RAMO_ROUP'
,'pre_QTD_PARCELAS_EMPRESTIMO_1'
,'pre_QTD_PARCELAS_EMPRESTIMO_2-6'
,'pre_QTD_PARCELAS_EMPRESTIMO_5'
,'ALVO_1_0']
##------------------------------------------------------------
## Separando em dados de treinamento e teste com Oversampling
##------------------------------------------------------------
# Separa em Treino e Teste
dataset_train = dataset.query("CONJUNTO == 'TRN'")
dataset_test = dataset.query("CONJUNTO == 'TST'")
# Seta Alvo
y_train = dataset_train['ALVO_1_0']
y_test = dataset_test['ALVO_1_0']
# Seta variáveis de entrada
X_train = dataset_train[cols_in]
X_test = dataset_test[cols_in]
#retirar alvo do 
del X_train['ALVO_1_0'] 
del X_test['ALVO_1_0'] 

#---------------------------------------------------------------------------
## Selecionando Atributos
#---------------------------------------------------------------------------
# feature extraction
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression(solver='newton-cg')
selected = RFE(model,45).fit(X_train, y_train)

print()
print('------------------     SELEÇÃO DE VARIÁVEIS     -------------------------------')
print("Num Features: %d" % selected.n_features_)
used_cols = []
for i in range(0, len(selected.support_)):
    if selected.support_[i]: 
        used_cols.append(X_train.columns[i]) 
        print('             -> {:30}     '.format(X_train.columns[i]))
    
print('-------------------------------------------------------------------------------')

X_train = X_train[used_cols]     # Carrega colunas de entrada selecionadas
X_test = X_test[used_cols]       # Carrega colunas de entrada selecionadas
#---------------------------------------------------------------------------
## Ajustando modelo - Aprendizado supervisionado  
#---------------------------------------------------------------------------
LogisticRegression = LogisticRegression(fit_intercept=True)
LogisticRegression.fit(X_train, y_train)
# Salvando os Modelos para implantar com outro código Python de ditribuição
du.SaveModel(LogisticRegression,'LogisticReg')

#Rede Neural com dados de treinamento
from sklearn.neural_network import MLPClassifier 
RNA = MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=True,
       epsilon=1e-08, hidden_layer_sizes=(3), learning_rate='constant',
       learning_rate_init=0.01, max_iter=5000,random_state=10,
       nesterovs_momentum=True, power_t=0.5, shuffle=True,
       solver='lbfgs', tol=0.0001)
RNA.fit(X_train, y_train)
# Salvando os Modelos para implantar com outro código Python de ditribuição
du.SaveModel(RNA,'RNA')

#---------------------------------------------------------------------------
## Previsão treinamento e teste
#---------------------------------------------------------------------------
y_pred_train_RLog = LogisticRegression.predict(X_train)
y_pred_test_RLog  = LogisticRegression.predict(X_test)
y_pred_train_RLog_R = LogisticRegression.predict_proba(X_train)[:,1]
y_pred_test_RLog_R  = LogisticRegression.predict_proba(X_test)[:,1]

y_pred_train_RNA_R = RNA.predict_proba(X_train)[:,1]
y_pred_test_RNA_R  = RNA.predict_proba(X_test)[:,1]
y_pred_train_RNA = RNA.predict(X_train)
y_pred_test_RNA  = RNA.predict(X_test)

#---------------------------------------------------------------------------
## Avaliação dos resultados
#---------------------------------------------------------------------------
print()
print('-------------------  Avalição de Desempenho  -------------------------------')
print('    RMSE Regressão Logística:',np.mean((y_pred_test_RLog_R - y_test) ** 2) ** 0.5)
print('    RMSE Redes Neurais      :',np.mean((y_pred_test_RNA_R - y_test) ** 2) ** 0.5)
print('    KS2  Regressão Logística:',du.KS2(y_test,y_pred_test_RLog_R))
print('    KS2  Redes Neurais      :',du.KS2(y_test,y_pred_test_RNA_R))
print('----------------------------------------------------------------------------')

#----------------------------------------------------------------------
## Montando um Data Frame (Matriz) com os resultados
#----------------------------------------------------------------------
# Conjunto de treinamento
df_train = pd.DataFrame(y_pred_train_RLog_R, columns=['REGRESSION_RLog'])
df_train['CLASSIF_RLog'] = y_pred_train_RLog
df_train['ALVO'] = [x for x in y_train]
df_train['CONJUNTO'] = 'TRT'
# Conjunto de test
df_test = pd.DataFrame(y_pred_test_RLog_R, columns=['REGRESSION_RLog'])
df_test['CLASSIF_RLog'] = y_pred_test_RLog
df_test['ALVO'] = [x for x in y_test]
df_test['CONJUNTO'] = 'TST' 
# Juntando Conjunto de Teste e Treinamento
df_total = pd.concat([df_test, df_train], sort = False)
## Exportando os dados para avaliação dos resultados em outra ferramenta
df_total.to_csv('resultado.csv')
print('--> DADOS EXPORTADOS')

