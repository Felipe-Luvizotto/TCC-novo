# --- ARQUIVO: core/treino_rf.py (Versão com Ponderação de Classe) ---

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def treinar_modelo_rf(X_treino, y_treino):
    print("DEBUG: Iniciando busca por hiperparâmetros do Random Forest com ponderação...")
    
    # --- CORREÇÃO: Adicionar 'class_weight' na criação do modelo ---
    # 'balanced' ajusta os pesos automaticamente com base na frequência das classes
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1')

    try:
        grid_search.fit(X_treino, y_treino)
        print(f"\nMelhores parâmetros para RF: {grid_search.best_params_}")
        print(f"Melhor F1-Score (validação cruzada): {grid_search.best_score_:.4f}")
        
        best_rf_model = grid_search.best_estimator_
        joblib.dump(best_rf_model, "modelo_rf.pkl")
        print("DEBUG: Melhor modelo Random Forest salvo.")
    except Exception as e:
        print(f"ERRO no GridSearch do RF: {e}. Treinando com padrão.")
        rf_model.fit(X_treino, y_treino)
        joblib.dump(rf_model, "modelo_rf.pkl")