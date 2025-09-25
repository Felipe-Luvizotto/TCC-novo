# --- ARQUIVO: core/models.py (Versão Corrigida) ---

import joblib
import os
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from core.model_lstm import LSTMModel

# --- CORREÇÃO: Atualizar o input_size do modelo LSTM para 7 ---
# O número de features agora é 7: Temp, Umid, Vento, Precip, Precip24h, Precip72h, Umid24h
INPUT_FEATURES = 7

# Inicializa as instâncias dos modelos como um fallback
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, n_estimators=100, random_state=42)
lstm_model = LSTMModel(input_size=INPUT_FEATURES, hidden_size=50) # <-- MUDANÇA AQUI
lstm_scaler = None

def carregar_modelos():
    """
    Tenta carregar os modelos pré-treinados e o scaler do LSTM.
    Se não existirem, usa as instâncias padrão (não treinadas).
    """
    global rf_model, xgb_model, lstm_scaler
    print("DEBUG [models.carregar_modelos]: Iniciando carregamento de modelos...")
    
    # Carregar modelo LSTM
    caminho_lstm = 'modelo_lstm.pth'
    if os.path.exists(caminho_lstm):
        try:
            # --- CORREÇÃO: Garante que o modelo carregado também tenha o tamanho certo ---
            lstm_model_carregado = LSTMModel(input_size=INPUT_FEATURES, hidden_size=50) # <-- MUDANÇA AQUI
            lstm_model_carregado.load_state_dict(torch.load(caminho_lstm))
            lstm_model_carregado.eval()
            lstm_model = lstm_model_carregado # Substitui o modelo global
            print("INFO [models.carregar_modelos]: 'modelo_lstm.pth' carregado com sucesso.")
        except Exception as e:
            print(f"ERRO [models.carregar_modelos]: Falha ao carregar 'modelo_lstm.pth': {e}")
    else:
        print("INFO [models.carregar_modelos]: 'modelo_lstm.pth' não encontrado.")

    # Carregar scaler do LSTM
    caminho_scaler_lstm = 'scaler_lstm.pkl'
    if os.path.exists(caminho_scaler_lstm):
        try:
            lstm_scaler = joblib.load(caminho_scaler_lstm)
            print("INFO [models.carregar_modelos]: 'scaler_lstm.pkl' carregado com sucesso.")
        except Exception as e:
            print(f"ERRO [models.carregar_modelos]: Falha ao carregar 'scaler_lstm.pkl': {e}")
    else:
        print("INFO [models.carregar_modelos]: 'scaler_lstm.pkl' não encontrado.")

    # Carregar modelo Random Forest
    caminho_rf = 'modelo_rf.pkl'
    if os.path.exists(caminho_rf):
        try:
            rf_model = joblib.load(caminho_rf)
            print(f"INFO [models.carregar_modelos]: Modelo RF carregado de '{caminho_rf}'.")
        except Exception as e:
            print(f"ERRO [models.carregar_modelos]: Falha ao carregar '{caminho_rf}': {e}")
    else:
        print(f"INFO [models.carregar_modelos]: '{caminho_rf}' não encontrado.")

    # Carregar modelo XGBoost
    caminho_xgb = 'modelo_xgb.pkl'
    if os.path.exists(caminho_xgb):
        try:
            xgb_model = joblib.load(caminho_xgb)
            print(f"INFO [models.carregar_modelos]: Modelo XGB carregado de '{caminho_xgb}'.")
        except Exception as e:
            print(f"ERRO [models.carregar_modelos]: Falha ao carregar '{caminho_xgb}': {e}")
    else:
        print(f"INFO [models.carregar_modelos]: '{caminho_xgb}' não encontrado.")

    print("DEBUG [models.carregar_modelos]: Fim do carregamento de modelos.")