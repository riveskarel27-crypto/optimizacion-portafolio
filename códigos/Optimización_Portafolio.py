#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:27:32 2026

@author: parisalvarez
"""


import numpy as np
from scipy.optimize import minimize

# =====================================================
# PROYECTO IDO
# Modelo de Markowitz: Portafolio de Mínima Varianza
# Método de optimización: SLSQP
# =====================================================

# 1. Instrumentos financieros
activos = ["DUK", "LQD", "KIMBERA.MX", "AC.MX"]

# 2. Vector de rendimientos esperados anuales (mu)
mu = np.array([
    0.12795459959881655,
    0.048374773852154065,
    0.08395689346754248,
    0.1517123938055161
])

# 3. Matriz de varianzas y covarianzas (Sigma)
Sigma = np.array([
    [0.02836283160471076, 0.0033342848638640388,
     0.002924452969247777, 0.0032471777671372742],

    [0.0033342848638640388, 0.005329421199429245,
     0.00276230671326149, 0.002260777429451611],

    [0.002924452969247777, 0.00276230671326149,
     0.059733581620689376, 0.018137591232738736],

    [0.0032471777671372742, 0.002260777429451611,
     0.018137591232738736, 0.05798654155415723]
])

# 4. Funcion objetivo: minimizar la mitad de la varianza
def funcion_objetivo(w):
    return 0.5 * w @ Sigma @ w

# Gradiente de la funcion objetivo
def gradiente(w):
    return Sigma @ w

# 5. Restriccion: la suma de los pesos debe ser 1
restricciones = [
    {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1,
        "jac": lambda w: np.ones_like(w)
    }
]

# 6. Restricciones de no negatividad
limites = [(0, 1) for _ in activos]

# 7. Pesos iniciales: misma proporcion para cada activo
w_inicial = np.ones(len(activos)) / len(activos)

# 8. Resolver el problema mediante SLSQP
resultado = minimize(
    fun=funcion_objetivo,
    x0=w_inicial,
    jac=gradiente,
    method="SLSQP",
    bounds=limites,
    constraints=restricciones,
    options={
        "maxiter": 1000,
        "ftol": 1e-12,
        "disp": False
    }
)

# 9. Mostrar resultados si la optimizacion fue exitosa
if resultado.success:
    pesos = resultado.x

    rendimiento = mu @ pesos
    varianza = pesos @ Sigma @ pesos
    volatilidad = np.sqrt(varianza)

    print("\nPORTAFOLIO OPTIMO DE MINIMA VARIANZA")
    print("-" * 45)

    for activo, peso in zip(activos, pesos):
        print(f"{activo:15s}: {peso * 100:10.4f}%")

    print("-" * 45)
    print(f"Suma de pesos:       {pesos.sum():.8f}")
    print(f"Rendimiento anual:   {rendimiento * 100:.4f}%")
    print(f"Varianza:            {varianza:.8f}")
    print(f"Volatilidad anual:   {volatilidad * 100:.4f}%")

    print("\nVerificacion de restricciones:")
    print("Suma de pesos igual a 1:",
          np.isclose(pesos.sum(), 1))
    print("Todos los pesos no negativos:",
          np.all(pesos >= -1e-8))

else:
    print("La optimizacion no convergio.")
    print("Mensaje:", resultado.message)


import pandas as pd
import numpy as np

if resultado.success:

    pesos = resultado.x
    rendimiento = mu @ pesos
    varianza = pesos @ Sigma @ pesos
    volatilidad = np.sqrt(varianza)

    # ==========================================
    # DOCUMENTO 1: RESULTADOS DE LOS ACTIVOS
    # ==========================================

    resultados_activos = pd.DataFrame({
        "Activo": activos,
        "Peso": pesos,
        "Porcentaje de inversion": pesos * 100,
        "Rendimiento esperado": mu,
        "Rendimiento esperado (%)": mu * 100
    })

    resultados_activos.to_csv(
        "resultados_portafolio.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==========================================
    # DOCUMENTO 2: INDICADORES DEL PORTAFOLIO
    # ==========================================

    resultados_generales = pd.DataFrame({
        "Concepto": [
            "Rendimiento esperado anual",
            "Varianza del portafolio",
            "Volatilidad anual"
        ],
        "Valor": [
            rendimiento,
            varianza,
            volatilidad
        ],
        "Porcentaje (%)": [
            rendimiento * 100,
            varianza * 100,
            volatilidad * 100
        ]
    })

    resultados_generales.to_csv(
        "indicadores_portafolio.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==========================================
    # MOSTRAR CONFIRMACION
    # ==========================================

    print("LOS DOS ARCHIVOS CSV SE CREARON CORRECTAMENTE.")

    print("\nDOCUMENTO 1: RESULTADOS DE LOS ACTIVOS")
    print(resultados_activos.to_string(index=False))

    print("\nDOCUMENTO 2: INDICADORES DEL PORTAFOLIO")
    print(resultados_generales.to_string(index=False))

else:
    print("La optimizacion no convergio.")
    print("Mensaje:", resultado.message)