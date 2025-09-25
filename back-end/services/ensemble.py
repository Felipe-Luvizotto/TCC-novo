# --- ARQUIVO: services/ensemble.py (Versão com Engenharia de Features) ---

import os, joblib, torch, numpy as np, sqlite3
from core.model_lstm import LSTMModel
from services.weather import get_weather_data
import json
import pandas as pd
from datetime import datetime
from core import models

def get_historical_features(lat: float, lon: float, conn):
    """
    Busca os dados mais recentes do DB para calcular as features de janela.
    """
    # Encontra o nome do município mais próximo pelas coordenadas
    query_municipio = "SELECT nome FROM municipios ORDER BY ((latitude - ?) * (latitude - ?)) + ((longitude - ?) * (longitude - ?)) ASC LIMIT 1"
    municipio_df = pd.read_sql_query(query_municipio, conn, params=(lat, lat, lon, lon))
    if municipio_df.empty:
        return None
    municipio = municipio_df.iloc[0]['nome']
    
    # Busca as últimas 72 horas de dados para esse município
    query_hist = "SELECT Precipitacao, Umidade FROM clima WHERE municipio = ? ORDER BY Data DESC, Hora DESC LIMIT 72"
    hist_df = pd.read_sql_query(query_hist, conn, params=(municipio,))
    
    if len(hist_df) < 72:
        print(f"AVISO: Menos de 72h de dados históricos para {municipio}. A previsão pode ser menos precisa.")

    # Calcula as features
    precip_acum_24h = hist_df['Precipitacao'].head(24).sum()
    precip_acum_72h = hist_df['Precipitacao'].sum()
    umid_media_24h = hist_df['Umidade'].head(24).mean()
    
    return precip_acum_24h, precip_acum_72h, umid_media_24h

def predict_ensemble(lat: float, lon: float):
    """
    Realiza a previsão de enchente usando features atuais e históricas.
    """
    weather_data = get_weather_data(lat, lon)
    if weather_data is None:
        return {"error": "Não foi possível obter dados climáticos atuais."}

    temp, humidity, wind_kph, precipitation = weather_data
    wind_ms = wind_kph / 3.6
    
    conn = sqlite3.connect('database.db')
    try:
        historical_features = get_historical_features(lat, lon, conn)
        if historical_features is None:
            return {"error": "Não foi possível encontrar dados históricos para esta localização."}
        
        precip_acum_24h, precip_acum_72h, umid_media_24h = historical_features
    finally:
        conn.close()

    # Prepara os dados para os modelos com TODAS as features na ordem correta
    data_for_models = np.array([
        temp, humidity, wind_ms, precipitation,
        precip_acum_24h, precip_acum_72h, umid_media_24h
    ]).reshape(1, -1)
    
    pred_rf = models.rf_model.predict_proba(data_for_models)[0][1]
    pred_xgb = models.xgb_model.predict_proba(data_for_models)[0][1]
    
    scaled_data_lstm = models.lstm_scaler.transform(data_for_models)
    data_lstm_tensor = torch.tensor(scaled_data_lstm.reshape(-1, 1, scaled_data_lstm.shape[1]), dtype=torch.float32)
    models.lstm_model.eval()
    with torch.no_grad():
        pred_lstm = models.lstm_model(data_lstm_tensor).item()

    ensemble_prediction = (pred_rf + pred_xgb + pred_lstm) / 3
    ensemble_prediction = min(1.0, max(0.0, ensemble_prediction))

    # Salva no histórico de previsões
    conn = sqlite3.connect('database.db')
    try:
        municipio_id = f"{lat},{lon}"
        data_hora_atual = datetime.now().isoformat()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO historico_previsao (municipio, data_hora, probabilidade) VALUES (?, ?, ?)",
                       (municipio_id, data_hora_atual, ensemble_prediction))
        conn.commit()
    finally:
        conn.close()

    return {
        "probabilidade": ensemble_prediction,
        "dados_atuais": { "Temperatura": temp, "Umidade": humidity, "Vento": wind_kph, "Precipitacao": precipitation }
    }

# A função predict_historical continua a mesma
def predict_historical(lat: float, lon: float, limit: int = 30):
    try:
        conn = sqlite3.connect('database.db')
        municipio_id = f"{lat},{lon}"
        df = pd.read_sql_query("SELECT data_hora, probabilidade FROM historico_previsao WHERE municipio = ? ORDER BY data_hora DESC LIMIT ?", conn, params=(municipio_id, limit))
        conn.close()
        if df.empty: return {"noData": True}
        df['data_hora'] = pd.to_datetime(df['data_hora'])
        history_list = df.sort_values(by='data_hora', ascending=True).apply(lambda row: {"timestamp": row['data_hora'].strftime('%d/%m %Hh'), "probability": row['probabilidade']}, axis=1).tolist()
        return history_list
    except Exception as e:
        print(f"ERRO: Falha ao buscar histórico de previsões: {e}")
        return {"erro": True, "detail": str(e)}