import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dividir_datos(data_path, test_size=0.2):
    """Divide los datos en entrenamiento y prueba usando solo numpy"""
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)
    
    # Mezclar los datos
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_samples = int(len(X) * test_size)
    
    X_train = X[indices[test_samples:]]
    y_train = y[indices[test_samples:]]
    X_test = X[indices[:test_samples]]
    y_test = y[indices[:test_samples]]
    
    # Guardar datos
    pd.DataFrame(np.hstack((X_train, y_train))).to_csv('dtrn.csv', index=False, header=False)
    pd.DataFrame(np.hstack((X_test, y_test))).to_csv('dtst.csv', index=False, header=False)
    
    return X_train, X_test, y_train, y_test

def calcular_vif(X):
    """Calcula el Factor de Inflación de Varianza para cada variable"""
    n_variables = X.shape[1]
    vif = np.zeros(n_variables)
    
    for i in range(n_variables):
        # Variable actual como target
        y = X[:, i].reshape(-1, 1)
        
        # Otras variables como features
        X_others = np.delete(X, i, axis=1)
        
        # Añadir columna de unos para el intercepto
        X_others = np.hstack((np.ones((X_others.shape[0], 1)), X_others))
        
        # Calcular coeficientes usando mínimos cuadrados
        beta = np.linalg.inv(X_others.T @ X_others) @ X_others.T @ y
        
        # Calcular R²
        y_pred = X_others @ beta
        ss_total = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        r_squared = 1 - (ss_res / ss_total)
        
        # Calcular VIF
        vif[i] = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
    
    return vif

def ajustar_modelo(X, y):
    """Ajusta un modelo de regresión lineal usando mínimos cuadrados"""
    # Añadir columna de unos para el intercepto
    X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # Calcular coeficientes usando la ecuación normal (X'X)^-1 X'y
    beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    
    return beta[0], beta[1:]  # intercepto, coeficientes

def calcular_indice_colineal(X, y):
    """
    Calcula el índice de colinealidad ponderado para cada variable
    
    I_j = sqrt(P * Q)
    P = VIF_j / mean(VIF)
    Q = var(beta_j) / mean(var(beta))
    """
    # Calcular VIF
    vif = calcular_vif(X)
    P = vif / np.mean(vif)
    
    # Ajustar modelo para obtener coeficientes
    intercept, beta = ajustar_modelo(X, y)
    
    # Calcular varianza de los coeficientes (usando matriz de covarianza)
    X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
    sigma_squared = np.sum((y - (intercept + X @ beta))**2) / (X.shape[0] - X.shape[1] - 1)
    cov_beta = sigma_squared * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    var_beta = np.diag(cov_beta)[1:]  # Excluir intercepto
    
    Q = var_beta / np.mean(var_beta)
    
    # Calcular índice
    I = np.sqrt(P * Q)
    
    return I

def seleccionar_variables(X, y, nombres_vars, tau=2):
    """
    Selecciona variables basado en el índice de colinealidad
    
    Args:
        X: matriz numpy con variables explicativas
        y: vector numpy con variable objetivo
        nombres_vars: lista con nombres de las variables
        tau: umbral para eliminar variables
        
    Returns:
        X_selected: matriz con variables seleccionadas
        nombres_selected: nombres de variables seleccionadas
        deleted_vars: lista de tuplas (nombre_var, indice) eliminadas
    """
    variables_actuales = nombres_vars.copy()
    indices_actuales = list(range(X.shape[1]))
    deleted_vars = []
    
    while True:
        if len(variables_actuales) == 0:
            break
            
        I = calcular_indice_colineal(X[:, indices_actuales], y)
        max_idx = np.argmax(I)
        max_val = I[max_idx]
        
        if max_val < tau:
            break
            
        # Eliminar variable con mayor índice
        deleted_var = variables_actuales.pop(max_idx)
        deleted_idx = indices_actuales.pop(max_idx)
        deleted_vars.append((deleted_var, max_val))
    
    return X[:, indices_actuales], variables_actuales, deleted_vars