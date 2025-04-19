import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calcular_rmse(y_real, y_pred):
    """Calcula el RMSE (Root Mean Squared Error)"""
    return np.sqrt(np.mean((y_real - y_pred)**2))

def calcular_r2(y_real, y_pred):
    """Calcula el R² (Coeficiente de Determinación)"""
    ss_total = np.sum((y_real - np.mean(y_real))**2)
    ss_res = np.sum((y_real - y_pred)**2)
    return 1 - (ss_res / ss_total)

def calcular_durbin_watson(residuales):
    """Calcula el estadístico de Durbin-Watson"""
    diff_res = np.diff(residuales.flatten())
    return np.sum(diff_res**2) / np.sum(residuales**2)

# 1. Cargar datos y modelo
try:
    dtst = pd.read_csv('dtst.csv', header=None).values
    X_test = dtst[:, :-1]
    y_test = dtst[:, -1].reshape(-1, 1)

    coefts = pd.read_csv('coefts.csv')
    selected_vars = pd.read_csv('selected_vars.csv')['variable'].tolist()

    # Obtener índices de las variables seleccionadas
    data_cols = pd.read_csv('dataset.csv').columns[:-1].tolist()
    selected_indices = [data_cols.index(var) for var in selected_vars if var in data_cols]
    X_test_selected = X_test[:, selected_indices]

    # 2. Hacer predicciones
    intercept = coefts[coefts['variable'] == 'intercept']['coef'].values[0]
    coef_values = coefts[coefts['variable'] != 'intercept']['coef'].values.reshape(-1, 1)

    y_pred = intercept + X_test_selected @ coef_values

    # 3. Calcular métricas
    rmse = calcular_rmse(y_test, y_pred)
    r2 = calcular_r2(y_test, y_pred)
    dw = calcular_durbin_watson(y_test - y_pred)

    # 4. Guardar resultados
    # Métricas
    pd.DataFrame({
        'metric': ['RMSE', 'R2', 'Durbin-Watson'],
        'value': [rmse, r2, dw]
    }).to_csv('metrica.csv', index=False)

    # Valores reales vs predichos
    pd.DataFrame({
        'real': y_test.flatten(),
        'pred': y_pred.flatten()
    }).to_csv('real_pred.csv', index=False)

    # 5. Crear gráficos
    # Configuración de estilo sin dependencia de seaborn
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.6

    # Figura 3: Real vs Estimado
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b-', label='Valor Real', linewidth=2, alpha=0.7)
    plt.plot(y_pred, 'r--', label='Valor Estimado', linewidth=2, alpha=0.7)
    plt.xlabel('Nro. Muestras', fontsize=12)
    plt.ylabel('Valor del Costo', fontsize=12)
    plt.title('Comparación: Valor Real vs. Estimado', fontsize=14)
    plt.legend(fontsize=10)
    plt.xticks(np.arange(0, len(y_test), 20))
    plt.tight_layout()
    plt.savefig('figure3.png', dpi=300)
    plt.close()

    # Figura 4: Residuales vs Estimado
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('Valor Estimado', fontsize=12)
    plt.ylabel('Residuales', fontsize=12)
    plt.title('Análisis de Residuales', fontsize=14)
    plt.tight_layout()
    plt.savefig('figure4.png', dpi=300)
    plt.close()

    print("Proceso completado exitosamente. Archivos generados:")
    print("- metrica.csv")
    print("- real_pred.csv")
    print("- figure3.png")
    print("- figure4.png")

except FileNotFoundError as e:
    print(f"Error: Archivo no encontrado - {e}")
    print("Asegúrate de tener todos los archivos necesarios:")
    print("- data.csv")
    print("- dtst.csv")
    print("- coefts.csv")
    print("- selected_vars.csv")
except Exception as e:
    print(f"Error inesperado: {e}")