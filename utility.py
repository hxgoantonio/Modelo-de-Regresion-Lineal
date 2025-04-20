import numpy as np

# Configurar semilla para reproducibilidad
np.random.seed(42)

def ajustar_modelo(X, y):
    """
    Ajusta un modelo de regresión lineal usando la matriz pseudoinversa
    vía descomposición de valores singulares
    """
    # Añadir columna de unos para el intercepto
    X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # Calcular coeficientes usando SVD para la pseudoinversa
    U, S, Vt = np.linalg.svd(X_with_intercept, full_matrices=False)
    
    # Crear matriz diagonal de valores singulares recíprocos
    S_inv = np.diag(1.0 / S)
    
    # Calcular pseudoinversa X+ = V * S_inv * U^T
    X_pinv = Vt.T @ S_inv @ U.T
    
    # Calcular coeficientes: beta = X+ * y
    beta = X_pinv @ y
    
    return beta[0], beta[1:]  # intercepto, coeficientes

def calcular_vif(X):
    """Calcula Factor de Inflación de Varianza para cada variable"""
    X = np.asarray(X, dtype=np.float64)
    n_features = X.shape[1]
    vif = np.zeros(n_features)

    for i in range(n_features):
        X_temp = np.delete(X, i, axis=1)
        y_temp = X[:, i]
        intercept, coef = ajustar_modelo(X_temp, y_temp)
        y_pred = intercept + X_temp @ coef
        r_squared = 1 - (np.sum((y_temp - y_pred)**2) / np.sum((y_temp - np.mean(y_temp))**2))
        vif[i] = 1. / (1. - r_squared) if r_squared < 1 else float('inf')

    return vif

def calcular_indice_colinealidad(X, y):
    """Calcula índice de colinealidad ponderado"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    vif = calcular_vif(X)
    intercept, coef = ajustar_modelo(X, y)
    
    X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
    U, S, Vt = np.linalg.svd(X_with_intercept, full_matrices=False)
    var_coef = np.diag(Vt.T @ np.diag(1.0 / (S**2)) @ Vt)[1:]
    
    P = vif / np.mean(vif)
    Q = var_coef / np.mean(var_coef)
    
    return np.sqrt(P * Q)