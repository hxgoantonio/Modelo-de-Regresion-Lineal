import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import ajustar_modelo, calcular_indice_colinealidad

def main():
    try:
        # 1. Cargar datos
        data = pd.read_csv('dataset.csv', header=0)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # 2. Dividir datos (80% train, 20% test)
        n_samples = X.shape[0]
        n_train = int(0.8 * n_samples)
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]
        
        # 3. Guardar conjuntos
        pd.DataFrame(np.hstack((X_train, y_train.reshape(-1, 1)))).to_csv('dtrn.csv', index=False, header=False)
        pd.DataFrame(np.hstack((X_test, y_test.reshape(-1, 1)))).to_csv('dtst.csv', index=False, header=False)
        
        # 4. Proceso de selección de variables (según pseudocódigo)
        tau = 2.0
        X_temp = X_train.copy()
        selected_vars = list(range(X_temp.shape[1]))
        deleted_vars = []
        deleted_idx_values = []
        
        while True:
            I = calcular_indice_colinealidad(X_temp, y_train)
            max_idx = np.argmax(I)
            max_val = I[max_idx]
            
            if max_val < tau:
                break
                
            deleted_vars.append(selected_vars[max_idx])
            deleted_idx_values.append(max_val)
            selected_vars.pop(max_idx)
            X_temp = np.delete(X_temp, max_idx, axis=1)
        
        # 5. Ajustar modelo final
        intercept, coef = ajustar_modelo(X_temp, y_train)
        
        # 6. Guardar resultados
        variables = ['Intercepto'] + ['X' + str(i+1) for i in selected_vars]
        pd.DataFrame({
            'Variable': variables,
            'Coeficiente': np.hstack(([intercept], coef))
            }).to_csv('coefts.csv', index=False)
        pd.DataFrame(['X'+str(i+1) for i in deleted_vars]).to_csv('deleted_vars.csv', index=False, header=False)
        pd.DataFrame(['X'+str(i+1) for i in selected_vars]).to_csv('selected_vars.csv', index=False, header=False)
        
        # 7. Gráficos
        # Figura1
        if deleted_vars:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(['X'+str(i+1) for i in deleted_vars], deleted_idx_values, color='#003366')
            ax.set_title('Variables Eliminadas: Índice de Colinealidad')
            ax.set_xlabel('Variable')
            ax.set_ylabel('Valor del Índice')
            
            ax.set_axisbelow(True)  # El grid va detrás de las barras
            ax.grid(True, which='both', axis='both', linestyle=':', color='gray', linewidth=0.3)  # Gris fuerte sin opacidad

            plt.savefig('figure1.png', bbox_inches='tight')
            plt.close()
        
        # Figura2
        if selected_vars:
            I_final = calcular_indice_colinealidad(X_temp, y_train)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(['X'+str(i+1) for i in selected_vars], I_final, color='#003366')  # Azul oscuro
            ax.set_title('Variables Seleccionadas: Índice de Colinealidad')
            ax.set_xlabel('Variable')
            ax.set_ylabel('Valor del Índice')
            
            ax.set_axisbelow(True)  # Cuadrícula detrás de las barras
            ax.grid(True, which='both', axis='both', linestyle=':', color='gray', linewidth=0.3)  # Cuadrícula fuerte y visible

            plt.savefig('figure2.png', bbox_inches='tight')
            plt.close()
            
            
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()