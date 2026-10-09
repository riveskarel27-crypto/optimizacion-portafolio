# ==============================================================================
# DESCARGA, LIMPIEZA, ESTADÍSTICA Y MATRICES DE MARKOWITZ
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf

# ------------------------------------------------------------------------------
# 0. CONFIGURACIÓN: Cambia aquí los 4 tickers y las fechas de consulta
# ------------------------------------------------------------------------------
TICKERS = ['DUK', 'LQD', 'KIMBERA.MX', 'AC.MX']
START_DATE = '2023-09-17'
END_DATE   = '2026-09-17'

# ------------------------------------------------------------------------------
# 1. Descarga, Limpieza Automática y Guardado de CSVs por Ticker
# ------------------------------------------------------------------------------
print('=' * 75)
print('1. DESCARGANDO DATOS DESDE YAHOO FINANCE Y GENERANDO ARCHIVOS CSV')
print('=' * 75)

precios = pd.DataFrame()

for ticker in TICKERS:
  print(f'Procesando {ticker}...')
  # Descarga de datos históricos desde Yahoo Finance
  df_raw = yf.download(
      ticker, start=START_DATE, end=END_DATE, progress=False
  )

  # Extracción directa del precio de cierre ('Close') evitando problemas de MultiIndex
  if isinstance(df_raw.columns, pd.MultiIndex):
    close_series = df_raw['Close'][ticker]
  else:
    close_series = df_raw['Close']

  # Limpieza y guardado del archivo CSV limpio individual ({ticker}.csv)
  df_clean = pd.DataFrame({'Close': close_series})
  df_clean.dropna(inplace=True)
  df_clean.to_csv(f'{ticker}.csv')

  # Asignación al DataFrame unificado de precios
  precios[ticker] = df_clean['Close']

# Eliminación de filas con valores nulos
precios.dropna(inplace=True)
print(
    f'-> Datos descargados e integrados correctamente ({len(precios)} observaciones diarias).\n'
)

# ------------------------------------------------------------------------------
# 2. Cálculo de Rendimientos Simples Discretos: Rt = (Pt - Pt-1) / Pt-1
# ------------------------------------------------------------------------------
rendimientos = precios.pct_change().dropna()

# ------------------------------------------------------------------------------
# 3. Análisis Estadístico Descriptivo y Prueba de Shapiro-Wilk
# ------------------------------------------------------------------------------
resumen_estadistico = []

for ticker in TICKERS:
  r = rendimientos[ticker]

  media_d = r.mean()
  vol_d = r.std(ddof=1)
  sesgo = r.skew()
  curtosis = r.kurtosis() + 3  # Curtosis real de Pearson (Normal = 3)

  # Prueba de hipótesis de Shapiro-Wilk
  sw_stat, p_value = stats.shapiro(r)

  resumen_estadistico.append({
      'Acción': ticker,
      'Rend. Diario Prom.': media_d,
      'Volatilidad Diaria': vol_d,
      'Sesgo (Skewness)': sesgo,
      'Curtosis': curtosis,
      'Estadístico SW': sw_stat,
      'P-Valor Shapiro-Wilk': p_value,
      '¿Es Normal? (alpha=0.05)': (
          'Sí' if p_value > 0.05 else 'No (Distribución No Normal)'
      ),
  })

df_stats = pd.DataFrame(resumen_estadistico)

print('=' * 75)
print(
    '2. TABLA DE PRUEBAS DE NORMALIDAD (SHAPIRO-WILK) Y MOMENTOS ESTADÍSTICOS'
)
print('=' * 75)
print(df_stats.to_string(index=False))

# ------------------------------------------------------------------------------
# 4. Construcción y Exportación de Insumos Anualizados (T = 252 días)
# ------------------------------------------------------------------------------
mu_anual = rendimientos.mean() * 252
sigma_anual = rendimientos.cov() * 252

# Exportación automática a CSV para los módulos subsecuentes
mu_anual.to_csv('mu_simple.csv', header=['Rendimiento_Anual'])
sigma_anual.to_csv('sigma_simple.csv')

print('\n' + '=' * 75)
print('3. VECTOR DE RENDIMIENTOS ESPERADOS ANUALIZADOS (mu_simple)')
print('=' * 75)
print(mu_anual.round(6))

print('\n' + '=' * 75)
print('4. MATRIZ DE VARIANZA-COVARIANZA ANUALIZADA (Sigma_simple)')
print('=' * 75)
print(sigma_anual.round(6))

# ------------------------------------------------------------------------------
# 5. Generación de Histogramas para la Sección Empírica (Matriz 2x2)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, ticker in enumerate(TICKERS):
  ax = axes[i]
  r = rendimientos[ticker]

  # Histograma empírico de rendimientos
  ax.hist(
      r,
      bins=40,
      density=True,
      alpha=0.6,
      color='skyblue',
      edgecolor='black',
      label='Rendimiento real',
  )

  # Curva normal teórica ajustada
  mu_r, std_r = r.mean(), r.std(ddof=1)
  x = np.linspace(r.min(), r.max(), 200)
  p = stats.norm.pdf(x, mu_r, std_r)
  ax.plot(x, p, 'r-', linewidth=2, label='Normal Teórica')

  # Indicador visual de la media
  ax.axvline(
      mu_r, color='black', linestyle='--', label=f'Media = {mu_r:.4f}'
  )

  ax.set_title(
      f'Distribución Muestral: {ticker}', fontsize=11, fontweight='bold'
  )
  ax.set_xlabel('Rendimiento Simple Diario', fontsize=9)
  ax.set_ylabel('Densidad', fontsize=9)
  ax.legend(fontsize=8)
  ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('histogramas_rendimientos_simples.png', dpi=300)
plt.show()