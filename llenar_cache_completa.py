import time
import json
import os
from balldontlie import BalldontlieAPI

# Configuración

API_KEY = "c4502107-deca-4812-84a2-bb689a2037d4" # Tu Key
client = BalldontlieAPI(api_key=API_KEY)

CACHE_FILE = "nba_stats_cache.json"
cache = {}

# Si existe, cargamos (pero idealmente bórralo antes de correr esto)
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
    except:
        cache = {}

print(f"--- REGENERANDO CACHÉ CON DATOS DE EQUIPO ---")

# IDs de equipos NBA (1 al 30)
for team_id in range(1, 31):
    str_id = str(team_id)
    
    # Si ya existe, asumimos que quieres actualizarlo o saltarlo.
    # Como los datos estaban mal, mejor sobrescribir si el puntaje es sospechosamente bajo (< 50)
    if str_id in cache:
        if cache[str_id]['pts'] > 50: 
            print(f"✅ Equipo {team_id} parece correcto ({cache[str_id]['pts']} pts). Saltando...")
            continue
        else:
            print(f"⚠️ Equipo {team_id} tiene datos de jugador ({cache[str_id]['pts']} pts). Corrigiendo...")

    print(f"⚡ Descargando Equipo {team_id}...", end="", flush=True)

    try:
        # Velocidad Plan de Pago
        time.sleep(1.2) 
        
        # 1. Traer últimos 10 juegos
        recent_games = client.nba.games.list(
            seasons=[2024], team_ids=[team_id], per_page=10
        )
        games_data = recent_games.data if hasattr(recent_games, 'data') else recent_games
        
        if not games_data:
            print(" [Sin juegos]")
            continue
            
        list_game_ids = [g.id for g in games_data]
        
        # 2. Traer stats detalladas
        stats_resp = client.nba.stats.list(game_ids=list_game_ids, per_page=100)
        stats_data = stats_resp.data if hasattr(stats_resp, 'data') else stats_resp

        # --- LÓGICA CORREGIDA: ACUMULADORES GLOBALES ---
        # Para porcentajes, sumamos todos los intentos y aciertos de TODOS los partidos
        global_fgm = 0; global_fga = 0
        global_fg3m = 0; global_fg3a = 0
        global_ftm = 0; global_fta = 0
        
        # Para promedios por partido (Puntos, Rebotes, Asistencias)
        # Usamos un diccionario para sumar por Game ID primero
        # games_totals = { 12345: {'pts': 110, 'reb': 45}, 12346: {'pts': 105...} }
        games_totals = {}

        for s in stats_data:
            # Validar que sea del equipo actual y haya jugado
            if hasattr(s, 'team') and s.team.id == team_id and s.min not in [None, "00"]:
                gid = s.game.id
                
                # Inicializar el partido si es nuevo
                if gid not in games_totals:
                    games_totals[gid] = {'pts': 0, 'reb': 0, 'ast': 0}
                
                # Sumar al total del PARTIDO (Team Totals)
                games_totals[gid]['pts'] += s.pts or 0
                games_totals[gid]['reb'] += s.reb or 0
                games_totals[gid]['ast'] += s.ast or 0
                
                # Acumular tiros para porcentajes globales
                global_fgm += s.fgm or 0; global_fga += s.fga or 0
                global_fg3m += s.fg3m or 0; global_fg3a += s.fg3a or 0
                global_ftm += s.ftm or 0; global_fta += s.fta or 0

        num_games = len(games_totals)
        
        if num_games > 0:
            # Calcular promedios basados en CANTIDAD DE PARTIDOS, no de jugadores
            total_pts = sum(g['pts'] for g in games_totals.values())
            total_reb = sum(g['reb'] for g in games_totals.values())
            total_ast = sum(g['ast'] for g in games_totals.values())

            stats_final = {
                # Porcentajes: Total Anotado / Total Intentado
                'fg_pct': round(global_fgm / global_fga, 3) if global_fga else 0,
                'ft_pct': round(global_ftm / global_fta, 3) if global_fta else 0,
                'fg3_pct': round(global_fg3m / global_fg3a, 3) if global_fg3a else 0,
                
                # Promedios: Total Acumulado / Número de Partidos
                'reb': round(total_reb / num_games, 1), 
                'ast': round(total_ast / num_games, 1), 
                'pts': round(total_pts / num_games, 1)
            }
            
            cache[str_id] = stats_final
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f, indent=4)
            
            print(f" [OK] -> Promedio: {stats_final['pts']} pts/partido")
        else:
            print(" [Datos insuficientes]")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if "429" in str(e): time.sleep(5)

print("\n--- PROCESO TERMINADO ---")