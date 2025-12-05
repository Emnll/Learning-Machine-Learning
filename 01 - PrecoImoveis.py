#%%
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
from sklearn.model_selection import train_test_split
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
import pandas as pd
import tarfile
import urllib

from sklearn.model_selection import StratifiedShuffleSplit
#%%
""" Download dos dados"""

#Faz o download do dataset que será analisado direto do github do criador do livro
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
# %%
# Cria um diretório chamado dataset, faz download do arquivo housing.tgz e extrai o arquivo housing.csv,
#  salvando tudo na pasta dataset
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# carrega os dados do arquivo housing
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
# %%
fetch_housing_data()

#%%
#Carrega o df pela função e mostra as primeiras cinco linhas do dataset
housing = load_housing_data()
housing.head()

# %%
# Mostra informações sobre cada atributo (coluna) do dataset
housing.info()
# %%
# Conta a quantidade de observações para cada tipo de categoria em "ocean_proximity"
# Vale notar que essa função é útil para variáveis categóricas

housing["ocean_proximity"].value_counts()
# %%
# Mostra informações como média, contagem, desvio padrão, mínimo, máximo e quartis para cada um dos atributos
housing.describe()

#%%
housing.hist(bins = 50, figsize = (20,15))
# %%
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
len(train_set)

#%%
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels = [1, 2, 3, 4, 5])
housing["income_cat"].hist()
# %%

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# %%
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
# %%
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
# %%
housing = strat_train_set.copy()
# %%
