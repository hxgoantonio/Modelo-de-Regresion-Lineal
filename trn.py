import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import dividir_datos, seleccionar_variables, ajustar_modelo, calcular_indice_colineal

# 1. Cargar y dividir datos
data = pd.read_csv('dataset.csv')
nombres_vars = data.columns[:-1].tolist()
X_train, X_test, y_train, y_test = dividir_datos('dataset.csv')

# 2. Seleccionar variables
X_selected, nombres_selected, deleted_vars = seleccionar_variables(
    X_train, y_train, nombres_vars
)

# 3. Ajustar modelo final con variables seleccionadas
intercept, coef = ajustar_modelo(X_selected, y_train)

# 4. Guardar resultados
# Coeficientes
coefts = pd.DataFrame({
    'variable': ['intercept'] + nombres_selected,
    'coef': [intercept[0]] + coef.flatten().tolist()
})
coefts.to_csv('coefts.csv', index=False)

# Variables eliminadas
if deleted_vars:
    deleted_df = pd.DataFrame(deleted_vars, columns=['variable', 'indice'])
    deleted_df.to_csv('deleted_vars.csv', index=False)
else:
    pd.DataFrame(columns=['variable', 'indice']).to_csv('deleted_vars.csv', index=False)

# Variables seleccionadas
pd.DataFrame({'variable': nombres_selected}).to_csv('selected_vars.csv', index=False)

# 5. Crear gráficos
# Figura 1: Variables eliminadas
if deleted_vars:
    plt.figure(figsize=(10, 6))
    variables, indices = zip(*deleted_vars)
    plt.bar(variables, indices)
    plt.xlabel('Variables Eliminadas')
    plt.ylabel('Índice de Colinealidad')
    plt.title('Variables Eliminadas vs Índice de Colinealidad')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figure1.png')
    plt.close()

# Figura 2: Variables seleccionadas
plt.figure(figsize=(10, 6))
I_selected = calcular_indice_colineal(X_selected, y_train)
plt.bar(nombres_selected, I_selected)
plt.xlabel('Variables Seleccionadas')
plt.ylabel('Índice de Colinealidad')
plt.title('Variables Seleccionadas vs Índice de Colinealidad')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figure2.png')
plt.close()