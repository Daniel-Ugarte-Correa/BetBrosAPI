import uvicorn
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
import shap
import traceback
import os
import random
import time
import json
from balldontlie import BalldontlieAPI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class FootballClient:
    def __init__(self, api_key):
        self.base_url = BASE_URL_FUTBOL
        self.headers = {
            # BORRA LA VARIABLE api_key Y PON TU LLAVE REAL AQUÍ DIRECTO:
            'x-apisports-key': "986da4fcfadcd3669adcf2763f4eb96e",
            'x-rapidapi-host': "v3.football.api-sports.io"
        }
# --- CONFIGURACIÓN DE APIS ---
# API FOOTBALL (v3.football.api-sports.io)
#API_KEY_FUTBOL = "986da4fcfadcd3669adcf2763f4eb96e"
#BASE_URL_FUTBOL = "https://v3.football.api-sports.io"

# BALLDONTLIE 
API_KEY_BALLDONTLIE = "c4502107-deca-4812-84a2-bb689a2037d4"


app = FastAPI()

# --- SISTEMA DE CACHÉ EN DISCO ---
CACHE_FILE = "nba_stats_cache.json"
CACHE_DATA = {}

def load_cache():
    """Carga datos guardados para no gastar API."""
    global CACHE_DATA
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                CACHE_DATA = json.load(f)
            print(f"INFO: Caché cargada ({len(CACHE_DATA)} equipos registrados).")
        except:
            CACHE_DATA = {}

def save_cache():
    """Guarda datos nuevos en disco."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(CACHE_DATA, f, indent=4)
    except Exception as e:
        print(f"WARN: No se pudo guardar caché: {e}")

# Iniciamos cargando la memoria
load_cache()

# --- CARGA DE MODELOS ---
model_football = None
X_train_football_df = None
explainer_football = None
model_basketball = None
X_train_basketball_df = None
explainer_basketball = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Fútbol
try:
    path_mod = os.path.join(BASE_DIR, 'modelo_futbol_final.pkl')
    path_dat = os.path.join(BASE_DIR, 'X_train_data_final.pkl')
    if os.path.exists(path_mod):
        model_football = joblib.load(path_mod)
        print("INFO: Modelo Fútbol cargado.")
    if os.path.exists(path_dat):
        X_train_football_df = joblib.load(path_dat)
        # Intentamos cargar SHAP si es compatible
        try:
            explainer_football = shap.TreeExplainer(model_football)
        except: pass
except Exception as e:
    print(f"WARN: No se cargó modelo fútbol: {e}")

# 2. Basketball
try:
    path_mod_basket = os.path.join(BASE_DIR, 'modelo_basket_v1.pkl')
    path_dat_basket = os.path.join(BASE_DIR, 'X_train_basket.pkl')
    
    if os.path.exists(path_mod_basket):
        model_basketball = joblib.load(path_mod_basket)
        print("INFO: Modelo Basket cargado.")
        
    if os.path.exists(path_dat_basket):
        X_train_basketball_df = joblib.load(path_dat_basket)
        try:
            explainer_basketball = shap.TreeExplainer(model_basketball)
        except: pass
except Exception as e:
    print(f"WARN: Error carga modelos basket: {e}")

# --- CLASES ---
class PredictionRequest(BaseModel):
    game_id: int
    deporte: str


class FootballClient:
    def __init__(self, api_key):
        self.base_url = BASE_URL_FUTBOL
        self.headers = {
            'x-apisports-key': api_key,
            'x-rapidapi-host': "v3.football.api-sports.io" 
        }
        self.team_stats_cache = {} 
        
        # MEJORA: Creamos una sesión con reintentos automáticos
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Configuración de reintentos (3 intentos, con espera entre ellos)
        retry_strategy = Retry(
            total=3,  # Número de intentos
            backoff_factor=1,  # Espera 1s, 2s, 4s...
            status_forcelist=[429, 500, 502, 503, 504], # Reintentar en estos errores
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get(self, endpoint, params=None):
        try:
            url = f"{self.base_url}/{endpoint}"
            # Usamos self.session.get en lugar de requests.get
            resp = self.session.get(url, params=params, timeout=10) # Timeout de 10s para no colgarse
            
            if resp.status_code == 200:
                data = resp.json()
                if "errors" in data and data["errors"]:
                    print(f"🛑 ERROR API ({endpoint}): {data['errors']}")
                return data
            
            print(f"Error HTTP {resp.status_code}: {resp.text}")
            return None

        except requests.exceptions.SSLError as ssl_err:
            print(f"⚠️ Error SSL (Red inestable): {ssl_err}")
            return None
        except Exception as e:
            print(f"Excepción de conexión: {e}")
            return None

    # --- MANTÉN TUS MÉTODOS DE LÓGICA IGUALES ---
    def get_todays_matches(self):
        today = date.today().strftime("%Y-%m-%d")
        data = self._get("fixtures", params={"date": today})
        
        if not data or not data.get('response'):
            print("No hay partidos hoy, buscando para mañana...")
            tomorrow = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
            data = self._get("fixtures", params={"date": tomorrow})
        return data

    def get_fixture_by_id(self, fixture_id):
        return self._get("fixtures", params={"id": fixture_id})

    def get_fixture_live_stats(self, fixture_id):
        return self._get("fixtures/statistics", params={"fixture": fixture_id})

    def get_team_season_stats(self, league_id, season, team_id):
        cache_key = f"{league_id}_{season}_{team_id}"
        if cache_key in self.team_stats_cache:
            return self.team_stats_cache[cache_key]

        data = self._get("teams/statistics", params={"league": league_id, "season": season, "team": team_id})
        if data and data.get('response'):
            self.team_stats_cache[cache_key] = data
            return data
        return None

    def get_last_matches_results(self, team_id):
        return self._get("fixtures", params={"team": team_id, "last": 5, "status": "FT"})
    
def calcular_probabilidades_trinaria(score_home, score_away):
    """
    Calcula probabilidades forzando la suma a 100% y potenciando
    el empate si el partido es parejo.
    """
    # 1. Suavizado (+1)
    s_h = score_home + 1.0
    s_a = score_away + 1.0
    
    # 2. Fuerza del Empate (AJUSTADA)
    # Antes era 0.35. Lo subimos a 0.55 para que, si los equipos son iguales,
    # el empate sea la opción más probable (aprox 35-36%).
    diferencia = abs(s_h - s_a)
    total_fuerza = s_h + s_a
    
    # La fórmula reduce el empate drásticamente si hay mucha diferencia
    peso_empate = (total_fuerza * 0.55) / (1 + (diferencia * 1.5))
    
    # 3. Totales y Porcentajes
    total_masa = s_h + s_a + peso_empate
    
    pct_h = (s_h / total_masa) * 100
    pct_d = (peso_empate / total_masa) * 100
    
    # TRUCO MAESTRO: El visitante es el resto. Así SIEMPRE suma 100.
    pct_a = 100.0 - (pct_h + pct_d)
    
    return pct_h, pct_d, pct_a

def obtener_recomendacion(prob, prediccion_texto, deporte):
    pct = prob * 100
    
    # Determinamos quién es el favorecido para el mensaje
    if "Local" in prediccion_texto or "Gana" in prediccion_texto and "Visitante" not in prediccion_texto:
        # Asumimos que si dice el nombre del equipo, lo sacamos del texto, o generalizamos
        equipo_favorecido = "el local"
    else:
        equipo_favorecido = "el visitante"

    # --- LÓGICA BASKETBALL 
    if deporte == "basketball":
        if pct >= 75:
            return {"titulo": "🔥 ALTA CONFIANZA", "mensaje": f"Ventaja muy clara para {equipo_favorecido}.", "color": "#4CAF50"}
        elif 60 <= pct < 75:
            return {"titulo": "⚖️ APUESTA MODERADA", "mensaje": f"Tendencia favorable para {equipo_favorecido}.", "color": "#FFC107"}
        else:
            return {"titulo": "🛑 RIESGO ALTO", "mensaje": f"Partido reñido. Precaución.", "color": "#F44336"}

    # --- LÓGICA FOOTBALL 
    elif deporte == "football":
        if pct >= 45:
            return {"titulo": "🔥 ALTA CONFIANZA", "mensaje": f"Altas probabilidades de éxito", "color": "#4CAF50"}
        elif 35 <= pct < 45:
            return {"titulo": "⚖️ APUESTA MODERADA", "mensaje": f"Probabilidad media de acierto", "color": "#FFC107"}
        else:
            # Menos de 35%
            return {"titulo": "🛑 RIESGO ALTO", "mensaje": f"Partido muy impredecible", "color": "#F44336"}
    
    # Default
    return {"titulo": "INFO", "mensaje": "Sin datos suficientes", "color": "#9E9E9E"}

def generar_simulacion_random(motivo):
    """Fallback si falla la API o modelos"""
    print(f"INFO: Generando random. Motivo: {motivo}")
    prob_local = random.uniform(0.3, 0.7)
    return {
        "prediccion": "Gana Local" if prob_local > 0.5 else "Gana Visitante",
        "explicacion": f"Simulación aleatoria ({motivo})",
        "probabilidades": {
            "local": f"{prob_local*100:.0f}%", 
            "visitante": f"{(1-prob_local)*100:.0f}%"
        },
        "recomendacion": {"titulo": "SIMULACIÓN", "mensaje": "Datos no disponibles.", "color": "#9E9E9E"}
    }



# --- LÓGICA BASKETBALL (OPTIMIZADA V6) ---

def get_nba_client():
    try:
        return BalldontlieAPI(api_key=API_KEY_BALLDONTLIE)
    except:
        return None

def mapear_features(home, away):
    """Convierte stats puras a features del modelo"""
    f = {}
    f['fg_pct_home'] = home['fg_pct']; f['ft_pct_home'] = home['ft_pct']
    f['fg3_pct_home'] = home['fg3_pct']; f['reb_home'] = home['reb']
    f['ast_home'] = home['ast']; f['plus_minus_home'] = home['pts'] - away['pts']
    
    f['fg_pct_away'] = away['fg_pct']; f['ft_pct_away'] = away['ft_pct']
    f['fg3_pct_away'] = away['fg3_pct']; f['reb_away'] = away['reb']
    f['ast_away'] = away['ast']; f['plus_minus_away'] = away['pts'] - home['pts']
    return f

def procesar_stats_con_cache(home_team_id, visitor_team_id):
    """
    Gestiona la descarga de datos respetando el límite de 5 req/min.
    Usa caché en disco para persistencia.
    """
    global CACHE_DATA
    
    str_home = str(home_team_id)
    str_visitor = str(visitor_team_id)
    
    # 1. ¿ESTÁN EN CACHÉ? (Si sí, retornamos rápido)
    if str_home in CACHE_DATA and str_visitor in CACHE_DATA:
        print(f"DEBUG: Usando caché local para {home_team_id} vs {visitor_team_id}")
        return mapear_features(CACHE_DATA[str_home], CACHE_DATA[str_visitor])
    
    # 2. SI NO, DESCARGAMOS (Lento y con cuidado)
    client = get_nba_client()
    if not client: return {}
    
    print(f"DEBUG: Descargando datos API para {home_team_id} vs {visitor_team_id}...")

    def descargar_equipo(team_id):
        # Verificar caché individual
        if str(team_id) in CACHE_DATA: return CACHE_DATA[str(team_id)]
        
        try:
            # Pausa obligatoria para no saturar (Límite 5/min = 1 cada 12 seg aprox)
            # Seremos un poco agresivos pero pausando entre llamadas
            time.sleep(2) 
            
            # Paso A: Juegos recientes
            recent_games = client.nba.games.list(
                seasons=[2024], team_ids=[team_id], per_page=10
            )
            games_data = recent_games.data if hasattr(recent_games, 'data') else recent_games
            if not games_data: return None
            
            list_game_ids = [g.id for g in games_data]
            
            time.sleep(2) # Otra pausa antes de pedir stats

            # Paso B: Stats
            stats_resp = client.nba.stats.list(game_ids=list_game_ids, per_page=100)
            stats_data = stats_resp.data if hasattr(stats_resp, 'data') else stats_resp
            
            # Paso C: Promediar
            totales = {'fgm':0, 'fga':0, 'fg3m':0, 'fg3a':0, 'ftm':0, 'fta':0, 'reb':0, 'ast':0, 'pts':0, 'games':0}
            for s in stats_data:
                if s.team.id == team_id and s.min not in [None, "00"]:
                    totales['fgm'] += s.fgm or 0; totales['fga'] += s.fga or 0
                    totales['fg3m'] += s.fg3m or 0; totales['fg3a'] += s.fg3a or 0
                    totales['ftm'] += s.ftm or 0; totales['fta'] += s.fta or 0
                    totales['reb'] += s.reb or 0; totales['ast'] += s.ast or 0
                    totales['pts'] += s.pts or 0
                    totales['games'] += 1
            
            if totales['games'] == 0: return None
            g = totales['games']
            
            return {
                'fg_pct': totales['fgm'] / totales['fga'] if totales['fga'] else 0,
                'ft_pct': totales['ftm'] / totales['fta'] if totales['fta'] else 0,
                'fg3_pct': totales['fg3m'] / totales['fg3a'] if totales['fg3a'] else 0,
                'reb': totales['reb'] / g, 'ast': totales['ast'] / g, 'pts': totales['pts'] / g
            }
        except Exception as e:
            if "429" in str(e) or "Too Many" in str(e):
                print("CRITICAL: Límite de API alcanzado (5 req/min).")
            else:
                print(f"Error descargando equipo {team_id}: {e}")
            return None

    # Intentamos descargar
    try:
        stats_h = descargar_equipo(home_team_id)
        if stats_h: 
            CACHE_DATA[str_home] = stats_h
            save_cache() # Guardamos progreso inmediatamente
        
        stats_v = descargar_equipo(visitor_team_id)
        if stats_v:
            CACHE_DATA[str_visitor] = stats_v
            save_cache()

        if stats_h and stats_v:
            return mapear_features(stats_h, stats_v)
        else:
            return {}

    except Exception as e:
        print(f"Error flujo caché: {e}")
        return {}



def interpretar_razon_futbol(feature_name, features_dict, home_name="Local", away_name="Visita", es_empate=False):
    """
    Genera narrativa. Si es_empate=True, destaca la paridad de los datos.
    CORREGIDO: Orden de variables para evitar UnboundLocalError.
    """
    # 1. Limpieza de nombres
    stat_base = feature_name.replace('_home', '').replace('_away', '').replace('rolling_avg_', '')
    
    val_h = features_dict.get(f"{stat_base}_home", 0)
    val_a = features_dict.get(f"{stat_base}_away", 0)
    
    # Fallback de búsqueda
    if val_h == 0 and val_a == 0:
        base_simple = stat_base.split('_')[-1]
        val_h = features_dict.get(f"{base_simple}_home", 0)
        val_a = features_dict.get(f"{base_simple}_away", 0)

    v_h_str = f"{val_h:.2f}"
    v_a_str = f"{val_a:.2f}"

    # ---------------------------------------------------------
    # CASO ESPECIAL: NARRATIVAS DE EMPATE / PARIDAD
    # ---------------------------------------------------------
    if es_empate:
        # CORRECCIÓN: Definimos la variable ANTES de usarla en el diccionario
        val_win = v_h_str 

        narrativas_empate = {
            'xGoals': [
                f"Duelo de fuerzas idénticas. Ambos equipos presentan un poder ofensivo muy similar ({val_win} xG), lo que sugiere un marcador cerrado.",
                f"La paridad es absoluta en ataque: tanto {home_name} como {away_name} promedian cerca de {val_win} goles esperados por juego."
            ],
            'shotsOnTarget': [
                f"Espejo táctico en el área. Ambos generan un volumen de tiros al arco casi idéntico ({val_win}), anulándose mutuamente.",
                f"El equilibrio defensivo es la clave: los dos equipos permiten y generan una cantidad similar de disparos ({val_win})."
            ],
            'corners': [
                f"Lucha territorial sin dueño. Las estadísticas de tiros de esquina son parejas ({val_win}), indicando que ninguno logra dominar el campo.",
                f"El juego por las bandas está muy equilibrado, con ambos equipos forzando un número similar de corners ({val_win})."
            ],
            'deep': [
                f"Las llegadas a línea de fondo son igual de frecuentes para ambos ({val_win}), pronosticando un choque muy trabado en mediocampo.",
                f"Ninguno logra romper la defensa rival con más frecuencia que el otro, mostrando métricas de profundidad calcadas."
            ]
        }
        
        import random
        if stat_base in narrativas_empate:
            return random.choice(narrativas_empate[stat_base])
        else:
            return f"La estadística de **{stat_base}** muestra una igualdad técnica notable ({v_h_str} vs {v_a_str}), lo que fundamenta la predicción de empate."

    # ---------------------------------------------------------
    # CASO NORMAL (HAY UN GANADOR CLARO)
    # ---------------------------------------------------------
    
    # Determinar quién gana la stat
    if val_h >= val_a:
        ganador, perdedor = home_name, away_name
        val_win, val_lose = v_h_str, v_a_str
    else:
        ganador, perdedor = away_name, home_name
        val_win, val_lose = v_a_str, v_h_str

    narrativas = {
        'xGoals': [
            f"El poder de fuego es evidente. {ganador} promedia {val_win} goles esperados, superior a la ofensiva de {perdedor} ({val_lose}).",
            f"La capacidad goleadora es clave: {ganador} genera ocasiones para {val_win} goles, superando a {perdedor} ({val_lose})."
        ],
        'shotsOnTarget': [
            f"Peligro constante: {ganador} dispara al arco {val_win} veces por partido, mucho más que {perdedor} ({val_lose}).",
            f"La precisión ofensiva favorece a {ganador} ({val_win} tiros al arco) frente a la timidez de {perdedor} ({val_lose})."
        ],
        'corners': [
            f"Presión constante: {ganador} embotella al rival con {val_win} corners por juego, superando a {perdedor} ({val_lose}).",
            f"El dominio territorial de {ganador} ({val_win} corners) inclina la balanza a su favor."
        ],
        'deep': [
            f"Juego vertical: {ganador} rompe líneas con {val_win} pases profundos, superando la creación de {perdedor}.",
            f"{ganador} llega a zona de peligro con más facilidad ({val_win}) que su rival."
        ]
    }

    import random
    nombres_bonitos = {
        'xGoals': "Goles Esperados (xG)", 'shotsOnTarget': "Tiros al Arco",
        'corners': "Tiros de Esquina", 'fouls': "Faltas", 'yellowCards': "Tarjetas"
    }

    if stat_base in narrativas:
        frase = random.choice(narrativas[stat_base])
    else:
        nombre_clean = nombres_bonitos.get(stat_base, stat_base)
        frase = f"Factor determinante: **{nombre_clean}**, donde {ganador} ({val_win}) supera a {perdedor} ({val_lose})."

    return frase


def interpretar_razon_basket(feature_name, features_dict, home_name="Local", away_name="Visita"):
    """
    Genera una narrativa natural y detallada explicando la estadística clave.
    Traduce abreviaturas técnicas (Ft Pct -> Tiros Libres).
    """
    # 1. Identificar la estadística base y los valores
    stat_base = feature_name.replace('_home', '').replace('_away', '')
    
    val_h = features_dict.get(f"{stat_base}_home", 0)
    val_a = features_dict.get(f"{stat_base}_away", 0)

    # 2. Formateo de valores (Porcentajes vs Enteros)
    es_porcentaje = 'pct' in stat_base
    if es_porcentaje:
        v_h_str = f"{val_h*100:.1f}%"
        v_a_str = f"{val_a*100:.1f}%"
    else:
        v_h_str = f"{val_h:.1f}"
        v_a_str = f"{val_a:.1f}"

    # 3. Determinar quién gana en esta stat
    if val_h >= val_a:
        ganador, perdedor = home_name, away_name
        val_win, val_lose = v_h_str, v_a_str
    else:
        ganador, perdedor = away_name, home_name
        val_win, val_lose = v_a_str, v_h_str

    # 4. DICCIONARIO DE NARRATIVAS (Agregamos ft_pct y turnover si hiciera falta)
    narrativas = {
        'ft_pct': [ # <--- ESTO FALTABA
            f"La seguridad desde la línea de castigo es vital. {ganador} capitaliza mejor las faltas con un {val_win} en tiros libres, superando el {val_lose} de {perdedor}.",
            f"Los 'puntos gratis' marcan la diferencia: {ganador} tiene una efectividad de {val_win} en libres, una clara ventaja sobre el {val_lose} de {perdedor}."
        ],
        'reb': [
            f"El control de los tableros es decisivo. {ganador} impone su físico promediando {val_win} rebotes, frente a los {val_lose} de {perdedor}.",
            f"La batalla en la pintura favorece a {ganador}, que captura {val_win} rebotes por partido, limitando las segundas oportunidades de {perdedor} ({val_lose})."
        ],
        'fg_pct': [
            f"La eficiencia de tiro inclina la balanza. {ganador} está más acertado con un {val_win} de campo, mientras que {perdedor} sufre con un {val_lose}.",
            f"La selección de tiro es clave: {ganador} anota el {val_win} de sus intentos, superando la efectividad de {perdedor} ({val_lose})."
        ],
        'fg3_pct': [
            f"El peligro desde el perímetro es el factor X. {ganador} castiga desde lejos con un {val_win} en triples, muy superior al {val_lose} de {perdedor}.",
            f"Nuestra IA destaca la puntería exterior de {ganador} ({val_win}) como un arma que la defensa de {perdedor} ({val_lose}) no podrá contener."
        ],
        'ast': [
            f"La circulación de balón define este duelo. {ganador} reparte {val_win} asistencias por noche, mostrando un juego más fluido que {perdedor} ({val_lose}).",
            f"El juego en equipo favorece a {ganador} ({val_win} asistencias), facilitando tiros más cómodos en comparación con {perdedor} ({val_lose})."
        ],
        'pts': [
            f"El poderío ofensivo es la razón principal. {ganador} anota con facilidad ({val_win} PPP), superando la media de {perdedor} ({val_lose}).",
            f"Simplemente, {ganador} tiene más dinamita en ataque, promediando {val_win} puntos frente a los {val_lose} de su rival."
        ],
        'plus_minus': [
            f"El rendimiento global (+/-) favorece claramente a {ganador}, indicando que dominan sus partidos de forma más consistente que {perdedor}.",
            f"Cuando los titulares de {ganador} están en cancha, el equipo suele sacar ventaja, un indicador estadístico superior al de {perdedor}."
        ]
    }

    # 5. Diccionario de Traducción de Seguridad (Fallback)
    # Si la stat no está arriba, usamos esto para que no salga "Ft Pct" nunca más.
    nombres_bonitos = {
        'ft_pct': "Porcentaje de Tiros Libres",
        'fg_pct': "Efectividad de Campo",
        'fg3_pct': "Efectividad de Triples",
        'reb': "Rebotes Totales",
        'ast': "Asistencias",
        'pts': "Puntos Anotados",
        'plus_minus': "Diferencial de Puntos",
        'turnover': "Pérdidas de Balón",
        'stl': "Robos de Balón",
        'blk': "Bloqueos"
    }

    import random
    
    if stat_base in narrativas:
        frase = random.choice(narrativas[stat_base])
    else:
        # Usamos el nombre bonito en español o lo limpiamos si no existe
        nombre_clean = nombres_bonitos.get(stat_base, stat_base.replace('_', ' ').title())
        frase = f"Un factor distintivo es la estadística de **{nombre_clean}**, donde {ganador} lidera con {val_win} sobre los {val_lose} de {perdedor}."

    return frase
# --- ENDPOINTS ---
LIGAS_TOP = [
    2,   # UEFA Champions League
    3,   # UEFA Europa League
    848, # UEFA Europa Conference League
    39,  # Premier League (Inglaterra)
    140, # La Liga (España)
    135, # Serie A (Italia)
    78,  # Bundesliga (Alemania)
    61,  # Ligue 1 (Francia)
    # --- Selecciones / Internacionales ---
    1,   # World Cup
    4,   # Euro Championship
    15,  # FIFA World Cup Qualification
    10,  # Friendlies (Amistosos Internacionales)
    5,   # UEFA Nations League
    9,   # Copa America
    13   # Copa Libertadores (Bonus: Muy popular)
]

@app.get("/partidos")
def get_partidos(deporte: str = "football"):
    
    if deporte == "football":
        try:
            fb_client = FootballClient(API_KEY_FUTBOL)
            data = fb_client.get_todays_matches()
            
            matches = []
            if data and 'response' in data:
                # 1. Filtramos: Solo guardamos si el ID de la liga está en nuestra lista TOP
                raw_matches = [m for m in data['response'] if m['league']['id'] in LIGAS_TOP]
                
                # 2. Ordenamos: Usamos el índice de la lista LIGAS_TOP para dar prioridad
                # (Los que están arriba en la lista LIGAS_TOP salen primero)
                raw_matches.sort(key=lambda x: LIGAS_TOP.index(x['league']['id']))

                for item in raw_matches: 
                    matches.append({
                        "game_id": item['fixture']['id'],
                        "league_id": item['league']['id'],
                        "nombreLiga": item['league']['name'],
                        "pais": item['league'].get('country', "Mundo"),
                        "equipoLocal": item['teams']['home']['name'],
                        "equipoVisitante": item['teams']['away']['name'],
                        "hora": item['fixture']['status']['short']
                    })
            
            if not matches:
                # Mensaje amigable si no juega ninguna liga top hoy
                return [{
                    "game_id": 0, "league_id": 0, "nombreLiga": "Sin Actividad Top", "pais": "Info",
                    "equipoLocal": "Hoy no juegan", "equipoVisitante": "las grandes ligas", "hora": "--"
                }]
                
            return matches

        except Exception as e:
            print(f"Error procesando partidos futbol: {e}")
            return [{
                "game_id": 0, "league_id": 0, "nombreLiga": "Error", "pais": "Error",
                "equipoLocal": "Error Backend", "equipoVisitante": "Revisar logs", "hora": "--"
            }]

    elif deporte == "basketball":
        client = get_nba_client()
        if client:
            try:
                today_str = date.today().strftime("%Y-%m-%d")
                # Intentamos obtener calendario
                # OJO: Esto gasta 1 petición. Si estás al límite, podría fallar.
                games = client.nba.games.list(dates=[today_str])
                data_games = games.data if hasattr(games, 'data') else games
                
                matches = []
                if data_games:
                    for g in data_games:
                        matches.append({
                            "game_id": g.id,
                            "nombreLiga": "NBA",
                            "equipoLocal": g.home_team.full_name,
                            "equipoVisitante": g.visitor_team.full_name,
                            "hora": g.status 
                        })
                    return matches
            except Exception as e:
                print(f"Error trayendo partidos: {e}")
        
        # Fallback si falla
        return [
            {"game_id": 101, "nombreLiga": "NBA (Sim)", "equipoLocal": "Lakers", "equipoVisitante": "Celtics", "hora": "20:00"},
            {"game_id": 102, "nombreLiga": "NBA (Sim)", "equipoLocal": "Warriors", "equipoVisitante": "Heat", "hora": "22:30"}
        ]

@app.post("/predecir")
def predecir(request: PredictionRequest):
    try:
        # --- FÚTBOL ---
        if request.deporte == "football":
            client = FootballClient(API_KEY_FUTBOL)
            
            # 1. Info básica del partido
            fixture_data = client.get_fixture_by_id(request.game_id)
            if not fixture_data or not fixture_data['response']:
                return generar_simulacion_random("Error ID partido")

            match_info = fixture_data['response'][0]
            league_id = match_info['league']['id']
            season_actual = match_info['league']['season']
            
            home_id = match_info['teams']['home']['id']; home_name = match_info['teams']['home']['name']
            away_id = match_info['teams']['away']['id']; away_name = match_info['teams']['away']['name']

            print(f"INFO: Analizando {home_name} vs {away_name} (Intento Temp {season_actual})")

            # -----------------------------------------------------------
            # FUNCIÓN INTELIGENTE DE EXTRACCIÓN DE DATOS
            # -----------------------------------------------------------
            def obtener_stats_robustas(l_id, s_year, t_id):
                # PASO 1: Intentar año actual (ej: 2025)
                # print(f"DEBUG: Intentando temporada {s_year}...") 
                r = client.get_team_season_stats(l_id, s_year, t_id)
                
                # Si funciona a la primera y trae datos, retornamos
                if r and r.get('response') and len(r['response']) > 0:
                    return r['response']
                
                # PASO 2: Si falló por CUALQUIER razón (Plan Free, error, vacío), probamos 2023
                # 2023 es el año seguro para cuentas gratis
                print(f"⚠️ No hay datos de {s_year} (Posible Plan Free). Cambiando a 2023...")
                r23 = client.get_team_season_stats(l_id, 2023, t_id)
                if r23 and r23.get('response') and len(r23['response']) > 0:
                    return r23['response']
                
                # PASO 3: Si 2023 falla, última oportunidad: 2022
                print(f"⚠️ 2023 falló. Probando 2022...")
                r22 = client.get_team_season_stats(l_id, 2022, t_id)
                if r22 and r22.get('response') and len(r22['response']) > 0:
                    return r22['response']
                
                return None

            # -----------------------------------------------------------
            # OBTENCIÓN DE DATOS
            # -----------------------------------------------------------
            stats_h = obtener_stats_robustas(league_id, season_actual, home_id)
            stats_a = obtener_stats_robustas(league_id, season_actual, away_id)

            features_ml = {}
            usar_ml = False
            detalle = ""
            year_used = "N/A"

            if stats_h and stats_a:
                try:
                    year_used = stats_h['league']['season']
                    
                    # --- FEATURE ENGINEERING (Transformar datos históricos a Inputs ML) ---
                    # Extraemos la "esencia" del equipo (sus promedios)
                    
                    # 1. Goles Promedio (Usamos esto como proxy de xGoals)
                    avg_goals_h = float(stats_h['goals']['for']['average']['total'] or 0)
                    avg_goals_a = float(stats_a['goals']['for']['average']['total'] or 0)
                    
                    # 2. Tiros al Arco Estimados (Un equipo suele tener 3 tiros al arco por cada gol)
                    est_sot_h = avg_goals_h * 3.5 
                    est_sot_a = avg_goals_a * 3.5
                    
                    # 3. Corners Estimados
                    est_corn_h = 4.0 + avg_goals_h
                    est_corn_a = 4.0 + avg_goals_a

                    # CREAMOS EL DICCIONARIO CON LOS NOMBRES EXACTOS QUE PIDE EL MODELO
                    features_ml = {
                        # --- Stats Directas ---
                        'xGoals_home': avg_goals_h,
                        'xGoals_away': avg_goals_a,
                        'shotsOnTarget_home': est_sot_h,
                        'shotsOnTarget_away': est_sot_a,
                        'corners_home': est_corn_h,
                        'corners_away': est_corn_a,
                        
                        # --- Stats Rellenadas (Promedios neutros para no romper el modelo) ---
                        'deep_home': 5.0, 'deep_away': 5.0,       # Pases profundos promedio
                        'fouls_home': 11.0, 'fouls_away': 11.0,   # Faltas promedio
                        'yellowCards_home': 2.0, 'yellowCards_away': 2.0,
                        'redCards_home': 0.1, 'redCards_away': 0.1,
                        
                        # --- Probabilidades de Casas de Apuestas (Neutrales 33%) ---
                        'homeProbability': 0.33, 
                        'drawProbability': 0.33, 
                        'awayProbability': 0.33,

                        # --- Rolling Averages (Usamos los mismos valores estimados) ---
                        'rolling_avg_xGoals_home': avg_goals_h,
                        'rolling_avg_xGoals_away': avg_goals_a,
                        'rolling_avg_corners_home': est_corn_h,
                        'rolling_avg_corners_away': est_corn_a,
                        'rolling_avg_shotsOnTarget_home': est_sot_h,
                        'rolling_avg_shotsOnTarget_away': est_sot_a,
                        'rolling_avg_fouls_home': 11.0,
                        'rolling_avg_fouls_away': 11.0
                    }
                    
                    detalle = f"Datos Históricos ({year_used}):\nLocal: {avg_goals_h} Goles/P\nVisita: {avg_goals_a} Goles/P"
                    usar_ml = True
                    print(f"DEBUG INPUT ML: Local (xG={avg_goals_h}, SoT={est_sot_h}) vs Visita (xG={avg_goals_a}, SoT={est_sot_a})")

                except Exception as e:
                    print(f"Error procesando datos: {e}")
            else:
                detalle = "Sin datos (Plan Free restringido)"
                print("ERROR: No se pudo obtener stats de 2025, 2023 ni 2022.")

            # -----------------------------------------------------------
            # EJECUCIÓN DEL MODELO ML
            # -----------------------------------------------------------
            pct_h, pct_d, pct_a = 34.0, 33.0, 33.0 # Valores base por si falla todo

            if usar_ml and model_football:
                try:
                    df_input = pd.DataFrame([features_ml])

                    # --- AGREGA ESTAS LÍNEAS AQUÍ (INICIO) ---
                    print("\n--- DIAGNÓSTICO DE COLUMNAS ---")
                    print(f"1. El modelo fue entrenado con estas columnas:\n{list(X_train_football_df.columns)}")
                    print(f"2. Tú le estás enviando estas columnas:\n{list(features_ml.keys())}")
                    print("-------------------------------\n")
                    # --- (FIN DEL CÓDIGO NUEVO) ---

                    # Rellenar columnas faltantes
                    for col in X_train_football_df.columns:
                        if col not in df_input.columns: df_input[col] = 0.0
                    df_input = df_input[X_train_football_df.columns]

                    # PREDECIR
                    probs = model_football.predict_proba(df_input)[0]
                    
                    # Asignación de Clases [0, 1, 2] -> [Visita, Empate, Local]
                    pct_a = probs[0] * 100
                    pct_d = probs[1] * 100
                    pct_h = probs[2] * 100
                    
                    print(f"DEBUG PROBS ML: A:{pct_a:.1f} D:{pct_d:.1f} H:{pct_h:.1f}")

                except Exception as ex:
                    print(f"Error ML Execution: {ex}")
                    usar_ml = False

            # -----------------------------------------------------------
            # NORMALIZACIÓN FINAL
            # -----------------------------------------------------------
            # Asegurar suma 100%
            total = pct_h + pct_d + pct_a
            if total > 0:
                pct_h = (pct_h/total)*100
                pct_d = (pct_d/total)*100
                pct_a = (pct_a/total)*100

            # Clamping Empate (Mínimo 15% para realismo)
            if pct_d < 15.0:
                diff = 15.0 - pct_d
                pct_d = 15.0
                pct_h -= diff/2
                pct_a -= diff/2

            # Definir Ganador
            max_prob = max(pct_h, pct_d, pct_a)
            prob_val = max_prob / 100.0
            
            if pct_h == max_prob: res = f"Gana {home_name}"
            elif pct_a == max_prob: res = f"Gana {away_name}"
            else: res = "Empate"

            # -----------------------------------------------------------
            # NARRATIVA INTELIGENTE FÚTBOL (SHAP + COHERENCIA)
            # -----------------------------------------------------------
            expl_text = "Análisis basado en el rendimiento histórico de la temporada."
            
            try:
                if explainer_football:
                    shap_vals = explainer_football.shap_values(df_input)
                    
                    # Indice según quién gana (0=Visita, 1=Empate, 2=Local)
                    idx_shap = 0
                    if pct_h == max_prob: idx_shap = 2 
                    elif pct_d == max_prob: idx_shap = 1
                    else: idx_shap = 0
                    
                    if isinstance(shap_vals, list): sv = shap_vals[idx_shap] 
                    else: sv = shap_vals
                    sv = np.array(sv).flatten()[:len(df_input.columns)]

                    # Ordenamos por importancia (impacto en el modelo)
                    indices_ordenados = np.argsort(np.abs(sv))[::-1]
                    
                    feat_seleccionada = None
                    es_empate = (res == "Empate") # Bandera clave

                    # BUCLE DE SELECCIÓN DE MEJOR ARGUMENTO
                    for i in indices_ordenados:
                        candidate = df_input.columns[i]
                        stat_base = candidate.replace('_home', '').replace('_away', '').replace('rolling_avg_', '')
                        
                        val_h = features_ml.get(f"{stat_base}_home", 0) or features_ml.get(f"rolling_avg_{stat_base}_home", 0)
                        val_a = features_ml.get(f"{stat_base}_away", 0) or features_ml.get(f"rolling_avg_{stat_base}_away", 0)
                        
                        # --- LÓGICA DE EMPATE ---
                        if es_empate:
                            # Si predecimos empate, buscamos stats donde la diferencia sea PEQUEÑA
                            # (Ej: 2.5 vs 2.4 es un buen argumento para empate)
                            diferencia = abs(val_h - val_a)
                            if diferencia < 1.0: # Umbral de "parecido"
                                feat_seleccionada = candidate
                                break
                        
                        # --- LÓGICA DE GANADOR (La de antes) ---
                        else:
                            es_local_ganador = (pct_h > pct_a)
                            if es_local_ganador and val_h >= val_a:
                                feat_seleccionada = candidate
                                break
                            elif not es_local_ganador and val_a >= val_h:
                                feat_seleccionada = candidate
                                break
                    
                    # Fallback si el bucle no encuentra nada
                    if not feat_seleccionada and len(indices_ordenados) > 0:
                        feat_seleccionada = df_input.columns[indices_ordenados[0]]

                    # Generar texto pasando la bandera es_empate
                    if feat_seleccionada:
                        expl_text = interpretar_razon_futbol(feat_seleccionada, features_ml, home_name, away_name, es_empate=es_empate)

            except Exception as e:
                print(f"Warn SHAP Futbol: {e}")
                expl_text = "Duelo muy cerrado basado en promedios históricos."

            return {
                "prediccion": res,
                # Usamos la nueva explicación narrativa
                "explicacion": f"{expl_text}\n\n--> {obtener_recomendacion(prob_val, res, 'football')['titulo']}",
                "probabilidades": {
                    "local": f"{pct_h:.0f}%", 
                    "empate": f"{pct_d:.0f}%", 
                    "visitante": f"{pct_a:.0f}%"
                },
                "recomendacion": obtener_recomendacion(prob_val, res, "football")
            }
        
        # --- BASKETBALL ---
        elif request.deporte == "basketball":
            if model_basketball is None: 
                return generar_simulacion_random("Modelo no cargado")

            features_dict = {}
            usando_datos_reales = False
            client = get_nba_client()

            # Intentar obtener datos reales
            if request.game_id > 1000 and client:
                try:
                    # 1. Obtener info básica (Gasta 1 Petición si no está cacheado por la librería interna)
                    # NOTA: Para ahorrar, asumimos IDs si fuera posible, pero necesitamos consultar el game
                    game_res = client.nba.games.get(game_id=request.game_id)
                    game = game_res.data if hasattr(game_res, 'data') else game_res
                    
                    home_id = game.home_team.id
                    visitor_id = game.visitor_team.id
                    
                    # 2. Procesar con Caché Inteligente
                    features_dict = procesar_stats_con_cache(home_id, visitor_id)
                    
                    if features_dict.get('reb_home', 0) > 0:
                        usando_datos_reales = True
                        
                except Exception as e:
                    print(f"Error obteniendo datos reales: {e}")

            # Fallback
            if not usando_datos_reales:
                return generar_simulacion_random("Datos insuficientes o API Limit (5/min)")

            # --- PREDICCIÓN ML ---
            features_df = pd.DataFrame([features_dict])
            
            # Rellenar columnas faltantes
            for col in X_train_basketball_df.columns:
                if col not in features_df.columns: features_df[col] = 0.0
            features_df = features_df[X_train_basketball_df.columns]
            
            # IMPRIMIR ESTO PARA VER EL ERROR
            print("\n--- DIAGNÓSTICO BASKET ---")
            print(f"1. Columnas que tienes guardadas (X_train): {len(features_df.columns)}")
            print(f"   Nombres: {list(features_df.columns)}")
            
            if hasattr(model_basketball, 'n_features_in_'):
                print(f"2. Columnas que el modelo ESPERA: {model_basketball.n_features_in_}")
            else:
                print("2. No se pudo leer n_features_in_ del modelo.")
            print("----------------------------\n")

            # Probabilidades
            probs = model_basketball.predict_proba(features_df)[0]
            idx = probs.argmax()
            res = "Gana Local" if idx == 1 else "Gana Visitante"
            rec = obtener_recomendacion(probs[idx], res, request.deporte)


            expl_text = "Análisis basado en estadísticas recientes."
            
            try:
                h_name = "Local"
                a_name = "Visita"
                if 'game' in locals() and hasattr(game, 'home_team'):
                     h_name = game.home_team.full_name
                     a_name = game.visitor_team.full_name

                # 1. Obtenemos valores SHAP
                shap_vals = explainer_basketball.shap_values(features_df)
                if isinstance(shap_vals, list): sv = shap_vals[idx]
                else: sv = shap_vals
                sv = np.array(sv).flatten()
                
                # Recorte de seguridad (el que hicimos antes)
                num_cols = len(features_df.columns)
                sv = sv[:num_cols]

                # 2. ORDENAMOS las características por importancia (de mayor a menor impacto)
                # np.argsort devuelve los índices ordenados de menor a mayor, por eso el [::-1]
                indices_ordenados = np.argsort(np.abs(sv))[::-1]

                # 3. BUSCAMOS LA MEJOR RAZÓN PARA EL GANADOR
                feat_seleccionada = None
                ganador_es_local = (idx == 1) # idx 1 es Local, 0 es Visita

                for i in indices_ordenados:
                    # Obtenemos el nombre de la columna candidata
                    candidate_feat = features_df.columns[i]
                    
                    # Extraemos la base (ej: 'reb_home' -> 'reb')
                    stat_base = candidate_feat.replace('_home', '').replace('_away', '')
                    
                    # Leemos los valores reales
                    val_h = features_dict.get(f"{stat_base}_home", 0)
                    val_a = features_dict.get(f"{stat_base}_away", 0)
                    
                    # VERIFICACIÓN DE COHERENCIA:
                    # ¿El equipo ganador es MEJOR en esta estadística?
                    if ganador_es_local and val_h >= val_a:
                        feat_seleccionada = candidate_feat
                        break # ¡Encontramos una razón válida!
                    elif not ganador_es_local and val_a >= val_h:
                        feat_seleccionada = candidate_feat
                        break # ¡Encontramos una razón válida!

                # 4. Fallback: Si por alguna razón el ganador es peor en TODO (muy raro),
                # usamos la variable más importante aunque sea negativa.
                if feat_seleccionada is None and len(indices_ordenados) > 0:
                    feat_seleccionada = features_df.columns[indices_ordenados[0]]

                # 5. Generamos el texto con la feature elegida
                if feat_seleccionada:
                    expl_text = interpretar_razon_basket(feat_seleccionada, features_dict, h_name, a_name)

            except Exception as e:
                print(f"Warn SHAP Logic: {e}")
                expl_text = "Análisis basado en la tendencia de la temporada."

            # Retorno Final
            return {
                "prediccion": res,
                "explicacion": f"{expl_text}\n\n--> {rec['titulo']}: {rec['mensaje']}",
                "probabilidades": {
                    "local": f"{probs[1]*100:.0f}%", 
                    "visitante": f"{probs[0]*100:.0f}%"
                },
                "recomendacion": rec
            }

    except Exception as e:
        traceback.print_exc()
        return {"prediccion": "Error", "explicacion": f"Fallo: {e}", "probabilidades": None, "recomendacion": None}

if __name__ == "__main__":
    # log_level="info" asegura que veas los mensajes de entrada/salida
    # access_log=True asegura que veas cada petición HTTP
    print("🚀 Servidor iniciado en http://0.0.0.0:8000")
    print("Esperando peticiones...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)
