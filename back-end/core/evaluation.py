# --- ARQUIVO: core/evaluation.py (Versão Final Corrigida) ---

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import json
import joblib # Importar joblib para carregar os modelos
import os

# --- CORREÇÃO: Remover a importação direta dos modelos globais ---
# from core.models import lstm_model, rf_model, xgb_model, lstm_scaler

def predict_lstm(data, lstm_model_local, lstm_scaler_local):
    """Prevê usando o modelo LSTM carregado localmente."""
    if lstm_scaler_local is None:
        print("AVISO: Scaler LSTM não carregado. Não é possível prever com LSTM.")
        return np.zeros(data.shape[0]) 
    
    scaled_data = lstm_scaler_local.transform(data)
    data_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(1)
    
    lstm_model_local.eval()
    with torch.no_grad():
        output = lstm_model_local(data_tensor)
        predictions = (output > 0.5).int().flatten().numpy()
    return predictions

def run_ensemble_evaluation(X_teste, y_teste):
    """
    Carrega os modelos do disco e avalia o desempenho de cada um e do ensemble.
    """
    print("DEBUG: Executando avaliação de modelos...")

    # --- CORREÇÃO: Carregar os modelos e o scaler diretamente do disco ---
    try:
        rf_model = joblib.load("modelo_rf.pkl")
        xgb_model = joblib.load("modelo_xgb.pkl")
        lstm_scaler = joblib.load("scaler_lstm.pkl")
        
        # Carregar o modelo LSTM
        from core.model_lstm import LSTMModel # Importa a classe do modelo
        # Determinar o input_size a partir dos dados de teste
        input_size = X_teste.shape[1]
        lstm_model = LSTMModel(input_size=input_size, hidden_size=50)
        lstm_model.load_state_dict(torch.load("modelo_lstm.pth"))
        lstm_model.eval()

        print("INFO: Todos os modelos e scaler foram carregados com sucesso para avaliação.")
    except FileNotFoundError as e:
        print(f"ERRO: Não foi possível carregar um dos arquivos de modelo para avaliação: {e}")
        return None
    except Exception as e:
        print(f"ERRO inesperado ao carregar modelos para avaliação: {e}")
        return None
    
    metrics = {}
    
    try:
        # Previsões
        rf_predictions = rf_model.predict(X_teste)
        xgb_predictions = xgb_model.predict(X_teste)
        # Passa os modelos carregados localmente para a função de previsão do LSTM
        lstm_predictions = predict_lstm(X_teste, lstm_model, lstm_scaler)

        # Ensemble
        ensemble_predictions = (rf_predictions + xgb_predictions + lstm_predictions >= 2).astype(int)

        # Cálculo de Métricas
        models_to_evaluate = {
            'Random_Forest': rf_predictions,
            'XGBoost': xgb_predictions,
            'LSTM': lstm_predictions,
            'Ensemble': ensemble_predictions
        }

        for name, predictions in models_to_evaluate.items():
            metrics[name] = {
                'accuracy': accuracy_score(y_teste, predictions),
                'precision': precision_score(y_teste, predictions, zero_division=0),
                'recall': recall_score(y_teste, predictions, zero_division=0),
                'f1_score': f1_score(y_teste, predictions, zero_division=0),
            }
        
        # Salvar as métricas em um arquivo JSON
        with open('evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        print("INFO: Métricas de avaliação salvas em 'evaluation_metrics.json'.")

        return metrics

    except Exception as e:
        print(f"ERRO: Falha na etapa de cálculo de métricas. Verifique os dados. Erro: {e}")
        return None

