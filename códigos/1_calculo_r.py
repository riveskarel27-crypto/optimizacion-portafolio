import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# 1. Unificación de CSVs de las 4 acciones

tickers = ['LMT', 'DIS', 'BMY', 'STT']

precios = pd.DataFrame()


for ticker in tickers:

    # Se buscan los archivos CSV en la carpeta actual

    df = pd.read_csv(f'{ticker}.csv', index_col='Date', parse_dates=True)

    precios[ticker] = df['Close']


# Limpiar posibles valores nulos

precios.dropna(inplace=True)


# 2. Rendimientos simples discretos Rt = (Pt - Pt-1) / Pt-1

rendimientos = precios.pct_change().dropna()


# 3. Análisis Estadístico Descriptivo y Prueba de Normalidad (Shapiro-Wilk)

resumen_estadistico = []

   

for ticker in tickers:

    r = rendimientos[ticker]

    media_d = r.mean()

    vol_d = r.std()

    sesgo = r.skew()

    curtosis = r.kurtosis() + 3  # SciPy/Pandas reportan exceso de curtosis (+3 da la curtosis real)

   

    # Prueba de Shapiro-Wilk para contraste de normalidad

    sw_stat, p_value = stats.shapiro(r)

   

    resumen_estadistico.append({

        'Acción': ticker,

        'Rend. Diario Prom.': media_d,

        'Volatilidad Diaria': vol_d,

        'Sesgo (Skewness)': sesgo,

        'Curtosis': curtosis,

        'Estadístico SW': sw_stat,

        'P-Valor Shapiro-Wilk': p_value,

        '¿Es Normal? (alpha=0.05)': 'Sí' if p_value > 0.05 else 'No (Distribución No Normal)'

    })


df_stats = pd.DataFrame(resumen_estadistico)

print("=== TABLA DE PRUEBAS DE NORMALIDAD (SHAPIRO-WILK) Y MOMENTOS ESTADÍSTICOS ===")

print(df_stats.to_string(index=False))


# 4. Construcción y Anualización de Matrices (Insumos Simples para Optimización)

mu_anual = rendimientos.mean() * 252

sigma_anual = rendimientos.cov() * 252


print("\n=== VECTOR DE RENDIMIENTOS ESPERADOS ANUALIZADOS (mu_simple) ===")

print(mu_anual)


print("\n=== MATRIZ DE VARIANZA-COVARIANZA ANUALIZADA (Sigma_simple) ===")

print(sigma_anual)


# 5. Generación de Histogramas para la Sección Empírica del Reporte

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes = axes.flatten()


for i, ticker in enumerate(tickers):

    ax = axes[i]

    r = rendimientos[ticker]

   

    # Histograma de los datos reales

    count, bins, ignored = ax.hist(r, bins=40, density=True, alpha=0.6, color='navy', edgecolor='black')

   

    # Curva Normal Teórica Ajustada

    mu_r, std_r = r.mean(), r.std()

    x = np.linspace(r.min(), r.max(), 100)

    p = stats.norm.pdf(x, mu_r, std_r)

    ax.plot(x, p, 'r-', linewidth=2, label='Normal Teórica')

   

    ax.set_title(f'Distribución de Rendimientos Simples: {ticker}')

    ax.set_xlabel('Rendimiento Simple Diario')

    ax.set_ylabel('Densidad')

    ax.legend()

    ax.grid(True, alpha=0.3)


plt.tight_layout()

plt.savefig('histogramas_rendimientos_simples.png', dpi=300)

plt.show() 