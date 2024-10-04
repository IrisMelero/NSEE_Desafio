import pandas as pd
import numpy as np
import gdown

# baixando e lendo arquivo e transformando-o em um dataframe
gdown.download("https://drive.google.com/uc?id=1KG9LxFsSeyEomMAtEi_m54QGBFjwcMNT", output=None, quiet=False)
df_base = pd.read_csv('pacigeral_jun24.csv')
# verificando
# print(df_base.shape)
# df_base.head()
# df_base.info()

# Item 1 - Selecionar pacientes com Topografia de pulmão (TOPOGRUP = C34)
df_item1 = df_base[df_base.TOPOGRUP.isin(['C34'])]
# print(df_item1.TOPOGRUP.value_counts())
# print(df_item1.shape)


# Item 2 - Selecionar pacientes com estado de Residência de São Paulo (UFRESID = SP)
df_item2 = df_item1[df_item1.UFRESID.isin(['SP'])]
# print(df_item2.UFRESID.value_counts())
# print(df_item2.shape)

# Item 3 - Selecionar pacientes com Base do Diagnóstico com Confirmaçãoo Microscópica (BASEDIAG = 3)
df_item3 = df_item2[df_item2.BASEDIAG.isin([3])]
# print(df_item3.BASEDIAG.value_counts())
# print(df_item3.shape)

# Item 4 - Retirar categorias 0, X e Y da coluna ECGRUP
df_item4 = df_item3[~df_item3.ECGRUP.isin(['0','X','Y'])]
# print(df_item4.ECGRUP.value_counts())
# print(df_item4.shape)

# Item 5 - Retirar pacientes que fizeram Hormonioterapia e TMO (HORMONIO = 1 e TMO = 1);
df_item5 = df_item4[~df_item4.HORMONIO.isin([1]) & ~df_item4.TMO.isin([1])]
# print(df_item5.HORMONIO.value_counts())
# print(df_item5.TMO.value_counts())
# print(df_item5.shape)

# Item 6 -  Selecionar pacientes com Ano de Diagnóstico até 2019 (ANODIAG ¡= 2019)
df_item6 = df_item5[(df_item5['ANODIAG'] <= 2019)]
# print(df_item6.ANODIAG.value_counts())
# print(df_item6.shape)

# Item 7 - Retirar pacientes com IDADE menor do que 20 anos
df_item7 = df_item6[~(df_item6['IDADE'] < 20)]
# print(df_item7.IDADE.value_counts().sort_index())
# print(df_item7.shape)

# Item 8 - 
# definindo dataframe do item 8
df_item8 = df_item7

# Converter as colunas para o formato datetime
df_item8['DTDIAG'] = pd.to_datetime(df_item8['DTDIAG'], errors='coerce')
df_item8['DTCONSULT'] = pd.to_datetime(df_item8['DTCONSULT'], errors='coerce')
df_item8['DTTRAT'] =  pd.to_datetime(df_item8['DTTRAT'], errors='coerce')

# Calcualr diferença de dias entre
    # diagnóstico e consulta
df_item8['CONSDIAG'] = (df_item8.DTDIAG - df_item8.DTCONSULT).dt.days
# print(df_item8.CONSDIAG.value_counts())
# print(df_item8.CONSDIAG)
    # tratamento e diagnóstico
df_item8['DIAGTRAT'] = (df_item8.DTTRAT - df_item8.DTDIAG).dt.days
# print(df_item8.DIAGTRAT.value_counts())
# print(df_item8.DIAGTRAT)
    # tratamento e consulta
df_item8['TRATCONS'] = (df_item8.DTTRAT - df_item8.DTCONSULT).dt.days
# print(df_item8.TRATCONS.value_counts())
# print(df_item8.TRATCONS)

# Codificando as colunas CONSDIAG, DIAGTRAT, TRATCONS
    # tratando campos não registrados - CONSDIAG não precisaria fazer essa verificação mas por desencargo ta aí
df_item8[['CONSDIAG', 'DIAGTRAT', 'TRATCONS']] = df_item8[['CONSDIAG', 'DIAGTRAT', 'TRATCONS']].fillna(-1)
# print(df_item8[['CONSDIAG', 'DIAGTRAT', 'TRATCONS']].isna().sum())

    # configurando CONSDIAG,  0 = até 30 dias; 1 = entre 31 e 60 dias; 2 = mais de 61 dias;
df_item8['CONSDIAG'] = [0 if consdiag <= 30 else 1 if consdiag <= 60 else 2 for consdiag in df_item8.CONSDIAG]
# print(df_item8['CONSDIAG'].value_counts())

    # configurando DIAGTRAT, 0 = até 60 dias; 1 = entre 61 e 90 dias; 2 = mais de 91 dias; 3 = não tratou (datas de tratamento vazias)
df_item8['DIAGTRAT'] = [3 if diagtrat < 0 else 0 if diagtrat <= 60 else 1 if diagtrat <= 90 else 2 for diagtrat in df_item8.DIAGTRAT]
# print(df_item8['DIAGTRAT'].value_counts())

    # configurando TRATCONS, 0 = até 60 dias; 1 = entre 61 e 90 dias; 2 = mais de 91 dias; 3 = não tratou (datas de tratamento vazias)
df_item8['TRATCONS'] = [3 if tratcons < 0 else 0 if tratcons <= 60 else 1 if tratcons <= 90 else 2 for tratcons in df_item8.TRATCONS]
# print(df_item8['TRATCONS'].value_counts())

# Item 9 - Extrair somente o número das colunas DRS e DRSINSTITU
    # Definindo df do item 9
df_item9 = df_item8

    # coluna DRS
drs_expand = df_item9.DRS.str.split(' ', expand=True)
df_item9['nDRS'] = drs_expand[1].astype(int) # definindo os dados como int
# print(df_item9['nDRS'].value_counts())

    # coluna DRSINSTITU
drsinst_expand = df_item9.DRSINST.str.split(' ', expand=True)
df_item9['nDRSINST'] = drsinst_expand[1].astype(int) # definindo os dados como int
# print(df_item9['nDRSINST'].value_counts())

# Item 10 - Criar a coluna binária de óbito, a partir da coluna ULTINFO, 
# onde as categorias 1 e 2 representam que o paciente está vivo 
# e as 3 e 4 representam o óbito por qualquer motivo;
    # Definindo df item 10
df_item10 = df_item9
    # Criando coluna óbito conforme valores de ULTINFO 
    # Considerando a preparação dos dados voltada para futura previsão de sobrevida de diferentespacientes, o evento de interesse (óbito) será armazenado por 1 e casos em que o paciente continua vivo serão armazenados com 0
df_item10['OBITO'] = np.where(df_item10['ULTINFO'].isin([3, 4]), 1, 0)
# print(df_item10[['OBITO','ULTINFO']])
# print(df_item10['OBITO'].value_counts())
# print(df_item10['ULTINFO'].value_counts())

# Item 11 - Retirar colunas
    # Definidno df item 11
df_item11 = df_item10
    # Definindo colunas a serem removidas
col_removidas = ['UFNASC', 'UFRESID', 'CIDADE', 'DTCONSULT', 'CLINICA' , 'DTDIAG', 'BASEDIAG', 'TOPOGRUP', 'DESCTOPO', 'DESCMORFO', 
                'T', 'N', 'M', 'PT', 'PN','PM', 'S' , 'G', 
                'LOCALTNM', 'IDMITOTIC', 'PSA', 'GLEASON', 'OUTRACLA',
                'META01', 'META02', 'META03', 'META04', 
                'DTTRAT', 'NAOTRAT', 'TRATAMENTO', 'TRATHOSP', 'TRATFANTES', 'TRATFAPOS', 
                'HORMONIO','TMO', 'NENHUMANT', 'CIRURANT', 'RADIOANT', 'QUIMIOANT', 'HORMOANT','TMOANT', 'IMUNOANT', 'OUTROANT', 'HORMOAPOS', 'TMOAPOS', 
                'DTULTINFO', 'CICI' , 'CICIGRUP', 'CICISUBGRU', 'FAIXAETAR', 'LATERALI', 'INSTORIG','RRAS', 'ERRO', 'DTRECIDIVA',
                'RECNENHUM', 'RECLOCAL', 'RECREGIO','RECDIST', 'REC01', 'REC02', 'REC03', 'REC04', 'CIDO',
                'HABILIT' , 'HABIT11' , 'HABILIT1' , 'CIDADEH', 'PERDASEG']
    # Removendo colunas especificadas
df_item11 = df_item11.drop(columns = col_removidas)
# print(df_item11.columns)