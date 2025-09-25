# --- ARQUIVO: core/treino_xgb.py (Versão Final sem Paralelismo) ---

import joblib
from core.models import xgb_model
from sklearn.model_selection import GridSearchCV
import numpy as np

def treinar_modelo_xgb(X_treino, y_treino):
    """
    Realiza o ajuste de hiperparâmetros (Grid Search) e treina o melhor
    modelo XGBoost.
    """
    print("DEBUG: Iniciando busca por hiperparâmetros do XGBoost...")

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [5, 10],
    }

    count_pos = np.sum(y_treino == 1)
    count_neg = np.sum(y_treino == 0)
    scale_pos_weight_value = 1
    if count_pos > 0 and count_neg > 0:
        scale_pos_weight_value = count_neg / count_pos
    print(f"XGBoost: Usando scale_pos_weight={scale_pos_weight_value:.2f}")

    estimator = xgb_model.set_params(scale_pos_weight=scale_pos_weight_value, use_label_encoder=False)

    # --- CORREÇÃO: Removido n_jobs=-1 para evitar o erro de serialização ---
    # O treinamento será mais lento, mas mais estável.
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=3,
        verbose=2,
        scoring='f1'
    )

    try:
        grid_search.fit(X_treino, y_treino)

        print("\n--- Resultados do Grid Search para XGBoost ---")
        print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
        print(f"Melhor F1-Score (validação cruzada): {grid_search.best_score_:.4f}")

        best_xgb_model = grid_search.best_estimator_
        joblib.dump(best_xgb_model, "modelo_xgb.pkl")
        print("DEBUG: Treinamento do melhor modelo XGBoost concluído e salvo.")

    except Exception as e:
        print(f"ERRO: Falha no ajuste de hiperparâmetros ou salvamento do XGBoost: {e}")
        print("INFO: Treinando com parâmetros padrão como fallback.")
        xgb_model.set_params(scale_pos_weight=scale_pos_weight_value)
        xgb_model.fit(X_treino, y_treino)
        joblib.dump(xgb_model, "modelo_xgb.pkl")