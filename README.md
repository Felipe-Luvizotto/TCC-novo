# Sistema de Previsão de Enchentes no Brasil com Machine Learning Ensemble

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.118-green.svg)
![React](https://img.shields.io/badge/React-18.x-blue.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red.svg)

Este repositório contém o código-fonte do Trabalho de Conclusão de Curso (TCC) para um sistema de previsão de enchentes em tempo real. A aplicação utiliza um ensemble de modelos de Machine Learning para calcular a probabilidade de ocorrência de enchentes em diversas localidades do Brasil, com base em dados históricos e informações climáticas atuais.

## 📜 Resumo

O projeto visa desenvolver uma solução completa, desde o processamento de grandes volumes de dados (mais de 40 milhões de registros) até a apresentação de uma previsão de risco em uma interface web interativa. A metodologia foi cuidadosamente desenhada para ser fiel à realidade dos dados, onde enchentes são eventos raros, utilizando técnicas de ponderação de classes e calibração de probabilidade para gerar previsões úteis e realistas.

## ✨ Principais Funcionalidades

*   **Visualização Geográfica:** Mapa interativo com a localização das estações meteorológicas do INMET.
*   **Previsão em Tempo Real:** Ao clicar em uma localidade, o sistema busca dados climáticos atuais e calcula a probabilidade de enchente para o dia.
*   **Dados Contextuais:** A previsão considera não apenas o clima atual, mas também o acúmulo de chuva e umidade das últimas 72 horas.
*   **Histórico de Previsões:** Gráfico que exibe a evolução da probabilidade de risco para a localidade selecionada.
*   **Transparência do Modelo:** Painel que exibe as métricas de performance (Acurácia, Precisão, Recall, F1-Score) do modelo ensemble e seus componentes.
*   **Orientações de Segurança:** Página informativa com dicas sobre como agir em caso de enchentes.

## 🏗️ Arquitetura

O sistema é dividido em duas partes principais:

1.  **Backend (Python/FastAPI):** Responsável por todo o pipeline de dados e Machine Learning.
    *   Processa e unifica dados históricos de enchentes (ANA) e dados climáticos (INMET).
    *   Realiza a engenharia de features para criar variáveis de contexto temporal.
    *   Treina um ensemble de modelos (Random Forest, XGBoost, LSTM).
    *   Calibra as probabilidades do modelo para fornecer resultados realistas.
    *   Expõe uma API RESTful para receber requisições de previsão.

2.  **Frontend (React):** Uma aplicação de página única (SPA) que consome a API do backend.
    *   Renderiza o mapa interativo usando a biblioteca `react-leaflet`.
    *   Envia as coordenadas da localidade selecionada para a API.
    *   Exibe os resultados da previsão (probabilidade, dados atuais, histórico) de forma clara e intuitiva.

## 🚀 Tecnologias Utilizadas

**Backend:**
*   **Linguagem:** Python 3.9+
*   **API Framework:** FastAPI
*   **Servidor:** Uvicorn
*   **Machine Learning:** Scikit-learn, XGBoost, PyTorch
*   **Manipulação de Dados:** Pandas, NumPy
*   **Banco de Dados:** SQLite3

**Frontend:**
*   **Framework:** React
*   **Mapa:** React-Leaflet, MarkerClusterGroup
*   **Gráficos:** Chart.js, react-chartjs-2
*   **Requisições HTTP:** Axios
*   **Estilização:** CSS puro

## ⚙️ Configuração e Instalação

Siga os passos abaixo para executar o projeto localmente.

### Pré-requisitos

*   Python 3.9 ou superior
*   Node.js e npm
*   Uma chave de API gratuita da [WeatherAPI](https://www.weatherapi.com/)

### Backend

1.  **Navegue até a pasta do backend:**
    ```bash
    cd back-end
    ```

2.  **Crie e ative um ambiente virtual:**
    *   No macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   No Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure a Chave da API de Clima:**
    Crie uma variável de ambiente com sua chave da WeatherAPI.
    *   No macOS/Linux:
        ```bash
        export WEATHER_API_KEY="SUA_CHAVE_API_AQUI"
        ```
    *   No Windows:
        ```bash
        set WEATHER_API_KEY="SUA_CHAVE_API_AQUI"
        ```

### Frontend

1.  **Abra um novo terminal** e navegue até a pasta do frontend:
    ```bash
    cd front-end
    ```

2.  **Instale as dependências:**
    ```bash
    npm install
    ```

## 🛠️ Ordem de Execução dos Scripts

Para que o sistema funcione, os scripts do backend devem ser executados na seguinte ordem. **Atenção:** Os passos de preparação e treinamento podem levar várias horas.

1.  **Limpeza (Opcional, mas recomendado para um início limpo):**
    Execute o script de limpeza para remover todos os arquivos gerados anteriormente.
    ```bash
    # No macOS/Linux
    chmod +x clean.sh
    ./clean.sh
    ```

2.  **1º - Preparação dos Dados:**
    Este script processará todos os dados climáticos (7GB+) e os salvará no banco de dados.
    ```bash
    python3 prepara_dados.py
    ```

3.  **2º - Treinamento dos Modelos:**
    Este é o passo mais demorado. Ele treinará os modelos com uma amostra de 4 milhões de registros.
    ```bash
    python3 treinamento_acelerado.py
    ```

4.  **3º - Calibração do Modelo:**
    Este script rápido treinará o calibrador de probabilidade.
    ```bash
    python3 calibrar_modelo.py
    ```

5.  **4º - Execução da Aplicação:**
    *   Inicie o servidor do backend (mantenha este terminal aberto):
        ```bash
        uvicorn main:app --reload
        ```
    *   Inicie a aplicação do frontend (em outro terminal):
        ```bash
        # Dentro da pasta front-end
        npm start
        ```

Acesse `http://localhost:3000` no seu navegador para ver a aplicação.

## 🔬 Metodologia de Machine Learning

Duas decisões metodológicas foram cruciais para o sucesso deste projeto:

1.  **Tratamento de Desbalanceamento de Classes:** Como enchentes são eventos raros, o dataset é naturalmente desbalanceado. Em vez de usar técnicas de reamostragem (que poderiam distorcer a realidade dos dados), optamos por usar **ponderação de classes** (`class_weight` e `scale_pos_weight`). Isso força os modelos a darem mais importância aos erros nos casos de enchente durante o treinamento, tornando-os mais sensíveis a esses eventos sem alterar a distribuição original dos dados.

2.  **Calibração de Probabilidade:** Modelos treinados com ponderação de classes tendem a produzir probabilidades "exageradas" (pontuações de confiança altas). Para transformar essas pontuações em probabilidades realistas, foi implementada uma etapa de pós-processamento utilizando **Regressão Isotônica**. Um calibrador foi treinado para mapear as previsões brutas do ensemble para probabilidades estatisticamente mais precisas, garantindo que uma previsão de "5%" realmente signifique um risco baixo, em vez de uma previsão irrealista de "40%".

## 🔮 Trabalhos Futuros

*   Fazer o deploy da aplicação em um serviço de nuvem (AWS, Google Cloud, Heroku).
*   Integrar novas fontes de dados, como níveis de rios e tipos de solo.
*   Automatizar o pipeline de retreinamento para que o modelo aprenda com novos dados periodicamente.
*   Explorar arquiteturas de modelos mais complexas.

## 👨‍💻 Autor

*   **Felipe Luvizotto** - [Felipe-Luvizotto](https://github.com/Felipe-Luvizotto)