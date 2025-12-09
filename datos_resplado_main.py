from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
import requests
from datetime import date

# 1. Carga del modelo
app = FastAPI()
model = joblib.load('modelo_futbol_v1.pkl')

# 2. Definición del modelo de datos de entrada
class MatchFeatures(BaseModel):
    xGoals_home: float
    xGoals_away: float
    shotsOnTarget_home: float
    shotsOnTarget_away: float
    deep_home: float
    deep_away: float
    fouls_home: float
    fouls_away: float
    corners_home: float
    corners_away: float
    yellowCards_home: float
    yellowCards_away: float
    redCards_home: float
    redCards_away: float
    homeProbability: float
    drawProbability: float
    awayProbability: float

# Ruta raíz para verificar que el servidor está activo
@app.get("/")
def read_root():
    return {"mensaje": "La API está funcionando correctamente!"}

# 3. Ruta para hacer predicciones
@app.post("/predecir")
def predecir(match: MatchFeatures):
    data = [list(match.dict().values())]
    prediction_numeric = model.predict(data)[0]
    resultado_map = {0: "Gana Visitante", 1: "Empate", 2: "Gana Local"}
    resultado = resultado_map.get(prediction_numeric, "Desconocido")
    return {"prediccion": resultado}

# 4. Ruta para obtener la lista de partidos REALES
@app.get("/partidos")
def get_partidos():
    # 1. Define los detalles de la API externa
    api_url = "https://free-api-live-football-data.p.rapidapi.com/football-get-matches-by-date"
    api_key = "e321eda217msha6715eef5c956c1p1b327bjsndfa24ec75765" 
    
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "free-api-live-football-data.p.rapidapi.com"
    }
    
    today_str = date.today().strftime("%Y%m%d")
    querystring = {"date": today_str}

    # 2. Haz la llamada a la API externa
    try:
        response = requests.get(api_url, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json()

        # 3. Procesa la respuesta (CÓDIGO CORREGIDO)
        partidos_reales = []
        # Accedemos a la lista que está en data['response']['matches']
        for partido_externo in data.get('response', {}).get('matches', []):
            partido = {
                # Accedemos a partido_externo['home']['name']
                "equipoLocal": partido_externo.get('home', {}).get('name', 'N/A'),
                # Accedemos a partido_externo['away']['name']
                "equipoVisitante": partido_externo.get('away', {}).get('name', 'N/A'),
                # Accedemos a partido_externo['time']
                "hora": partido_externo.get('time', 'N/A')
            }
            partidos_reales.append(partido)
        
        return partidos_reales

    except requests.exceptions.RequestException as e:
        return {"error": f"Error al contactar la API externa: {e}"}