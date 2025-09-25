# --- ARQUIVO: core/treino_lstm.py (Versão Corrigida) ---

import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from core.model_lstm import LSTMModel

def treinar_modelo_lstm(X_treino, y_treino, columns):
    print("DEBUG: Iniciando treinamento do modelo LSTM...")

    try:
        df_treino = pd.DataFrame(X_treino, columns=columns)
        df_treino['Enchente'] = y_treino

        df_treino.dropna(inplace=True)

        if df_treino.empty:
            print("AVISO: Dataset vazio para LSTM. Não é possível treinar.")
            return

        X_treino_clean = df_treino.drop('Enchente', axis=1).values.astype(np.float32)
        y_treino_clean = df_treino['Enchente'].values.astype(np.float32)

        if X_treino_clean.shape[0] == 0:
            print("AVISO: Nenhuma amostra válida para treinar o LSTM.")
            return

        # Normalização dos dados
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_treino_scaled = scaler.fit_transform(X_treino_clean)
        joblib.dump(scaler, 'scaler_lstm.pkl')

        X_treino_tensor = torch.tensor(X_treino_scaled, dtype=torch.float32).unsqueeze(1)
        y_treino_tensor = torch.tensor(y_treino_clean, dtype=torch.float32).unsqueeze(1)

        # --- CORREÇÃO: O input_size agora é o número de colunas que chegam ---
        input_size = X_treino_scaled.shape[1]
        print(f"DEBUG [LSTM]: Treinando modelo com input_size = {input_size}")

        hidden_layer_size = 50
        num_epochs = 100
        learning_rate = 0.001
        
        model = LSTMModel(input_size=input_size, hidden_size=hidden_layer_size) # <-- MUDANÇA AQUI
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for i in range(num_epochs):
            optimizer.zero_grad()
            y_pred = model(X_treino_tensor)
            loss = loss_function(y_pred, y_treino_tensor)
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(f"  LSTM - Época {i+1}/{num_epochs}, Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), 'modelo_lstm.pth')
        print("DEBUG: Treinamento do modelo LSTM concluído e salvo.")

    except Exception as e:
        print(f"ERRO: Falha ao treinar ou salvar o modelo LSTM: {e}")