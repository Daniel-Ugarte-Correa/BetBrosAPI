import pandas as pd








games_df = pd.read_csv('archive/games.csv')
leagues_df = pd.read_csv('archive/leagues.csv')
teams_df = pd.read_csv('archive/teams.csv')
teamstats_df = pd.read_csv('archive/teamstats.csv')









df.isnull().sum()










print("Columnas en games_df:")
print(games_df.columns)
print("\n--------------------------\n") # Un separador para que sea más claro
print("Columnas en teamstats_df:")
print(teamstats_df.columns)







# --- Paso 2 Corregido: Unir estadísticas del equipo LOCAL ---
# Unimos 'games_df' con 'teamstats_df'
# La llave es: la ID del equipo local en 'games_df' debe coincidir con la ID del equipo en 'teamstats_df'
# Y también deben coincidir en el 'gameID'
partidos_con_stats = pd.merge(
    games_df, 
    teamstats_df, 
    left_on=['gameID', 'homeTeamID'], 
    right_on=['gameID', 'teamID']
)
# --- Paso 3 Corregido: Unir estadísticas del equipo VISITANTE ---
# Ahora unimos el resultado anterior de nuevo con 'teamstats_df', pero para el equipo visitante
# Usamos 'suffixes' para que las columnas no se llamen igual (ej: 'goals_home' y 'goals_away')
partidos_completos = pd.merge(
    partidos_con_stats,
    teamstats_df,
    left_on=['gameID', 'awayTeamID'],
    right_on=['gameID', 'teamID'],
    suffixes=('_home', '_away')
)
# --- Inspección del Resultado ---
print("Merge exitoso. Estas son las primeras 5 filas:")
partidos_completos.head()
























import numpy as np

# Crear la columna 'winner' con 3 posibles valores
conditions = [
    partidos_completos['homeGoals'] > partidos_completos['awayGoals'],  # Condición para victoria local
    partidos_completos['homeGoals'] == partidos_completos['awayGoals']  # Condición para empate
]
outcomes = [2, 1] # Resultados correspondientes: 2 para victoria local, 1 para empate

# np.select crea la columna. El valor por defecto (si no se cumple ninguna condición) es 0
partidos_completos['winner'] = np.select(conditions, outcomes, default=0)

# Verificamos la nueva columna y los resultados para ver si tiene lógica
print("Se ha creado la columna 'winner'. Así se ven los resultados:")
print(partidos_completos[['homeGoals', 'awayGoals', 'winner']].head(10))




















import matplotlib.pyplot as plt
import seaborn as sns

# Seleccionamos solo las columnas numéricas para el análisis de correlación
numeric_df = partidos_completos.select_dtypes(include=np.number)

# Creamos el mapa de calor para ver las relaciones
plt.figure(figsize=(20, 15)) # Hacemos la figura grande para que sea legible
heatmap = sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm') # Pon annot=True si quieres ver los números, pero puede ser ilegible
heatmap.set_title('Mapa de Calor de Correlaciones', fontdict={'fontsize':18}, pad=12)
plt.show()

















# Definimos la lista de columnas que usaremos como 'pistas'
features = [
    'xGoals_home', 'xGoals_away',
    'shotsOnTarget_home', 'shotsOnTarget_away',
    'deep_home', 'deep_away',
    'fouls_home', 'fouls_away',
    'corners_home', 'corners_away',
    'yellowCards_home', 'yellowCards_away',
    'redCards_home', 'redCards_away',
    'homeProbability', 'drawProbability', 'awayProbability'
]

# Creamos nuestras variables X e y
X = partidos_completos[features]
y = partidos_completos['winner']

# Imprimimos el tamaño para verificar
print("Forma de nuestras features (X):", X.shape)
print("Forma de nuestro target (y):", y.shape)





















from sklearn.model_selection import train_test_split

# Dividimos los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, # 20% para prueba
    random_state=42 # Para que la división sea siempre la misma y reproducible
)






















from sklearn.ensemble import RandomForestClassifier

# 1. Creamos el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 2. Entrenamos el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

print("¡Modelo entrenado exitosamente!")




























from sklearn.metrics import accuracy_score, classification_report

# Hacemos predicciones en el set de prueba
predictions = model.predict(X_test)

# Calculamos la precisión
accuracy = accuracy_score(y_test, predictions)
print(f"La precisión del modelo es: {accuracy * 100:.2f}%")

# Vemos un reporte más detallado
print("\nReporte de Clasificación:")
print(classification_report(y_test, predictions, target_names=['Gana Visitante (0)', 'Empate (1)', 'Gana Local (2)']))








































<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:background="@color/background_green_gray"
    tools:context=".HistorialActivity">

    <androidx.appcompat.widget.Toolbar
        android:id="@+id/toolbar_historial"
        android:layout_width="0dp"
        android:layout_height="?attr/actionBarSize"
        android:background="@color/almost_black"
        android:elevation="4dp"
        android:theme="@style/ThemeOverlay.AppCompat.Dark.ActionBar"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent"
        app:title="Mi Historial" />

    <androidx.cardview.widget.CardView
        android:id="@+id/card_balance"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:layout_marginStart="16dp"
        android:layout_marginTop="16dp"
        android:layout_marginEnd="16dp"
        app:cardCornerRadius="8dp"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toBottomOf="@id/toolbar_historial">

        <androidx.constraintlayout.widget.ConstraintLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:padding="16dp">

            <TextView
                android:id="@+id/tv_label_balance"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text="Balance Apuestas"
                android:textSize="22sp"
                android:textStyle="bold"
                app:layout_constraintStart_toStartOf="parent"
                app:layout_constraintTop_toTopOf="parent" />

            <TextView
                android:id="@+id/tv_balance_total"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:textColor="@android:color/holo_green_dark"
                android:textSize="28sp"
                android:textStyle="bold"
                app:layout_constraintStart_toStartOf="parent"
                app:layout_constraintTop_toBottomOf="@id/tv_label_balance"
                tools:text="$5,00" />

            <TextView
                android:id="@+id/tv_label_apuestas_balance"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:layout_marginTop="8dp"
                android:text="Apuestas"
                android:textStyle="bold"
                app:layout_constraintStart_toStartOf="parent"
                app:layout_constraintTop_toBottomOf="@id/tv_balance_total" />

            <TextView
                android:id="@+id/tv_apuestas_balance"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:textStyle="bold"
                app:layout_constraintStart_toStartOf="parent"
                app:layout_constraintTop_toBottomOf="@id/tv_label_apuestas_balance"
                tools:text="$25,00" />

            <TextView
                android:id="@+id/tv_label_retornos_balance"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text="Retornos"
                android:textStyle="bold"
                app:layout_constraintBaseline_toBaselineOf="@+id/tv_label_apuestas_balance"
                app:layout_constraintStart_toStartOf="@+id/guideline_center" />

            <TextView
                android:id="@+id/tv_retornos_balance"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:textStyle="bold"
                app:layout_constraintBaseline_toBaselineOf="@+id/tv_apuestas_balance"
                app:layout_constraintStart_toStartOf="@+id/guideline_center"
                tools:text="$30,00" />

            <androidx.constraintlayout.widget.Guideline
                android:id="@+id/guideline_center"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:orientation="vertical"
                app:layout_constraintGuide_percent="0.5" />

        </androidx.constraintlayout.widget.ConstraintLayout>
    </androidx.cardview.widget.CardView>

    <androidx.cardview.widget.CardView
        android:id="@+id/card_mayor_ganancia"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:layout_marginTop="16dp"
        app:cardCornerRadius="8dp"
        app:layout_constraintEnd_toEndOf="@id/card_balance"
        app:layout_constraintStart_toStartOf="@id/card_balance"
        app:layout_constraintTop_toBottomOf="@id/card_balance">

        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="vertical"
            android:padding="20dp">

            <TextView
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text="Mayor Ganancia"
                android:layout_marginBottom="18dp"
                android:textSize="20sp"
                android:textStyle="bold" />

            <TextView
                android:id="@+id/tv_partido_mayor_ganancia"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:textSize="18sp"
                android:textStyle="bold"
                tools:text="Local - Visitante" />

            <LinearLayout
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:orientation="horizontal"
                android:layout_marginTop="18dp">

                <TextView android:id="@+id/tv_apuesta_mayor_ganancia" android:textStyle="bold" android:layout_width="0dp" android:layout_weight="1" android:layout_height="wrap_content" tools:text="Apuestas $5,00" />
                <TextView android:id="@+id/tv_resultado_mayor_ganancia" android:textStyle="bold"  android:layout_width="0dp" android:layout_weight="1" android:layout_height="wrap_content" android:gravity="center" tools:text="Empate"/>
                <TextView android:id="@+id/tv_retorno_mayor_ganancia" android:textStyle="bold" android:layout_width="0dp" android:layout_weight="1" android:layout_height="wrap_content" android:gravity="end" tools:text="Retornos $30,00"/>
            </LinearLayout>
        </LinearLayout>
    </androidx.cardview.widget.CardView>

    <androidx.cardview.widget.CardView
        android:id="@+id/card_mayor_perdida"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:layout_marginTop="12dp"
        app:cardCornerRadius="8dp"
        app:layout_constraintEnd_toEndOf="@id/card_balance"
        app:layout_constraintStart_toStartOf="@id/card_balance"
        app:layout_constraintTop_toBottomOf="@id/card_mayor_ganancia">

        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="vertical"
            android:padding="20dp">

            <TextView
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text="Mayor Pérdida"
                android:layout_marginBottom="18dp"
                android:textSize="20sp"
                android:textStyle="bold" />

            <TextView
                android:id="@+id/tv_partido_mayor_perdida"
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:textStyle="bold"
                android:textSize="18sp"
                tools:text="Local - Visitante" />

            <LinearLayout
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:orientation="horizontal"
                android:layout_marginTop="18dp">
                <TextView android:id="@+id/tv_apuesta_mayor_perdida" android:textStyle="bold" android:layout_width="0dp" android:layout_weight="1" android:layout_height="wrap_content" tools:text="Apuestas $20,00" />
                <TextView android:id="@+id/tv_resultado_mayor_perdida" android:textStyle="bold" android:layout_width="0dp" android:layout_weight="1" android:layout_height="wrap_content" android:gravity="center" tools:text="Gana Local"/>
                <TextView android:id="@+id/tv_retorno_mayor_perdida" android:textStyle="bold" android:layout_width="0dp" android:layout_weight="1" android:layout_height="wrap_content" android:gravity="end" tools:text="Retornos $0,00"/>
            </LinearLayout>
        </LinearLayout>
    </androidx.cardview.widget.CardView>

    <androidx.recyclerview.widget.RecyclerView
        android:id="@+id/recycler_view_historial"
        android:visibility="gone"
        tools:visibility="visible"
        android:layout_width="0dp"
        android:layout_height="0dp"
        android:layout_marginTop="16dp"
        app:layout_constraintBottom_toTopOf="@+id/bottom_navigation_historial"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toBottomOf="@id/card_mayor_perdida" />

    <com.google.android.material.bottomnavigation.BottomNavigationView
        android:id="@+id/bottom_navigation_historial"
        android:layout_width="0dp"
        android:layout_height="wrap_content"
        android:background="@color/almost_black"
        app:itemIconSize="@dimen/footer_icon_size"
        app:itemIconTint="@null"
        app:itemTextAppearanceActive="@style/BottomNavTextStyle"
        app:itemTextAppearanceInactive="@style/BottomNavTextStyle"
        app:itemTextColor="@android:color/white"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:menu="@menu/historial_bottom_nav_menu" />

</androidx.constraintlayout.widget.ConstraintLayout>