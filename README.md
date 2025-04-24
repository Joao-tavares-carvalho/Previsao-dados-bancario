# Previsão-dados-bancario
Usando modelos de inteligência artificial para prever o tipo de cliente, utilizando simulação de dados bancários, tendo o Streamlit como interface.

CAPTURA DE TELA DO APP
![streamli_1](https://github.com/user-attachments/assets/a8d6a836-8068-42ac-ad7e-40ba6a94a3ee)

PRINCIPAIS IMPORTACÕES
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

