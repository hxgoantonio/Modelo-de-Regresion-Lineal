import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import ajustar_modelo

def calcular_rmse(y_real, y_pred):
    return np.sqrt(np.mean((y_real - y_pred)**2))

def calcular_r2(y_real, y_pred):
    ss_total = np.sum((y_real - np.mean(y_real))**2)
    ss_res = np.sum((y_real - y_pred)**2)
    return 1 - (ss_res / ss_total)

def calcular_durbin_watson(residuals):
    diff = np.diff(residuals)
    return np.sum(diff**2) / np.sum(residuals**2)

def main():
    try:
        # 1. Cargar coeficientes y variables
        coeffs_df = pd.read_csv('coefts.csv')
        intercept = coeffs_df[coeffs_df['Variable'] == 'Intercepto']['Coeficiente'].values[0]
        coef = coeffs_df[coeffs_df['Variable'] != 'Intercepto']['Coeficiente'].values
        selected_vars = pd.read_csv('selected_vars.csv', header=None).values.flatten()
        selected_indices = [int(var[1:])-1 for var in selected_vars]
        
        # 2. Cargar y preparar datos de prueba
        test_data = pd.read_csv('dtst.csv', header=None).values
        X_test = test_data[:, selected_indices]
        y_test = test_data[:, -1]
        
        # 3. Calcular predicciones
        y_pred = intercept + X_test @ coef
        
        # 4. Calcular y guardar métricas
        rmse = calcular_rmse(y_test, y_pred)
        r2 = calcular_r2(y_test, y_pred)
        residuals = y_test - y_pred
        dw = calcular_durbin_watson(residuals)
        
        pd.DataFrame([[rmse, r2, dw]], columns=['RMSE', 'R2', 'Durbin-Watson']).to_csv('metrica.csv', index=False)
        pd.DataFrame({'Real': y_test, 'Predicted': y_pred}).to_csv('real_pred.csv', index=False)
        
        # 5. Gráfico figure3.png (Real vs Estimados)
        plt.figure(figsize=(12, 6))
        muestras = np.arange(len(y_test))
        
        plt.plot(muestras, y_test, 'b-', linewidth=1.5, label='Real Values')
        plt.plot(muestras, y_pred, 'r--', linewidth=1.5, label='Estimated Value')
        
        plt.ylim(-15, 20)
        plt.yticks([-15, -10, -5, 0, 5, 10, 15, 20])
        plt.xticks(np.arange(0, len(y_test)+1, 20))
        
        plt.xlabel('Nro. Muestras', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.title('Real vs Estimados', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.savefig('figure3.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Gráfico figure4.png (Residuals)
        plt.figure(figsize=(10, 5))
        plt.scatter(y_pred, residuals, alpha=0.6, edgecolor='k', linewidth=0.5)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
        plt.xlabel('Estimated-Y values', fontsize=11)
        plt.ylabel('Residuals', fontsize=11)
        plt.title('Residuals vs Estimados', fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig('figure4.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error en tst.py: {e}")
        raise

if __name__ == '__main__':
    main()