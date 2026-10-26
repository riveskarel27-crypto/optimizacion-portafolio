"""
==========================================================================
¿Nuestro portafolio está dentro de la región factible?
==========================================================================
GLOSARIO RAPIDO (lo único que necesitas saber)
 - Portafolio ........ una forma de repartir tu dinero entre los 4 activos.
 - Pesos ............. cuánto % del dinero va a cada activo (suman 100%, nunca negativos).
 - Rendimiento ....... cuánto esperas ganar al año con ese reparto.
 - Riesgo ............ qué tanto sube y baja ese rendimiento (volatilidad).
 - Región factible ... TODOS los pares (riesgo, rendimiento) que se pueden lograr
                       con algún reparto. Lo que cae fuera NO se puede armar.
 - Frontera eficiente  el borde de arriba-izquierda de la región: para cada nivel de
                       riesgo, el reparto que más rinde. Nadie debería quedar debajo.
 - GMVP .............. el reparto de MENOR riesgo posible (el de Paris).
 - Tangente .......... el reparto con mejor rendimiento por unidad de riesgo (Sharpe).
 - Diversificar ...... mezclar activos que no suben y bajan juntos para bajar el riesgo.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "mysql-connector-python", "numpy", "pandas", "scipy", "matplotlib"])

# ------------------------------ AJUSTES ----------------------------------
ACTIVOS = ["DUK", "LQD", "KIMBERA.MX", "AC.MX"]
TASA_LIBRE_RIESGO = 0.04      # lo que daría una inversión sin riesgo 
USAR_MYSQL = True           # cámbialo a True cuando tengas MySQL listo (ver el final del archivo)
MYSQL = dict(host="localhost", user="root", password="contraseña")

# --------------------------- PASO 1: LOS DATOS ----------------------------
#  - RENDIMIENTO[i] = cuánto se espera que gane cada activo al año
#  - COV = tabla de "qué tanto se mueven juntos" los activos (diagonal = su propio riesgo al cuadrado)
RENDIMIENTO = np.array([0.12795459959881655, 0.048374773852154065,
                        0.08395689346754248, 0.1517123938055161])
COV = np.array([
    [0.02836283160471076, 0.0033342848638640388, 0.002924452969247777, 0.0032471777671372742],
    [0.0033342848638640388, 0.005329421199429245, 0.00276230671326149, 0.002260777429451611],
    [0.002924452969247777, 0.00276230671326149, 0.059733581620689376, 0.018137591232738736],
    [0.0032471777671372742, 0.002260777429451611, 0.018137591232738736, 0.05798654155415723]])
N = len(ACTIVOS)

# --------------------- PASO 2: FUNCIONES BÁSICAS --------------------------
def rendimiento_de(pesos):   # promedio ponderado de los rendimientos de cada activo
    return RENDIMIENTO @ pesos

def riesgo_de(pesos):        # riesgo del reparto, tomando en cuenta cómo se mueven juntos
    return np.sqrt(pesos @ COV @ pesos)

def repartir_con_menor_riesgo(rendimiento_objetivo=None):
    """Busca los pesos con MENOR riesgo (sin pesos negativos y sumando 100%).
    Si se da un rendimiento_objetivo, exige además ganar exactamente eso."""
    reglas = [{"type": "eq", "fun": lambda w: w.sum() - 1}]          # que sumen 100%
    if rendimiento_objetivo is not None:
        reglas.append({"type": "eq", "fun": lambda w: RENDIMIENTO @ w - rendimiento_objetivo})
    r = minimize(lambda w: 0.5 * w @ COV @ w, np.ones(N) / N, method="SLSQP",
                 bounds=[(0, 1)] * N, constraints=reglas, options={"ftol": 1e-14, "maxiter": 1000})
    return r.x

def mayor_riesgo_posible(rendimiento_objetivo):
    """El borde DERECHO de la región. Truco matemático: el reparto más arriesgado
    para un rendimiento dado siempre usa como máximo 2 activos, así que basta
    probar todas las parejas de activos."""
    mayor = 0.0
    for i, j in combinations(range(N), 2):
        if RENDIMIENTO[i] == RENDIMIENTO[j]:
            continue
        # ¿qué parte del dinero en i (y el resto en j) da justo ese rendimiento?
        parte_i = (rendimiento_objetivo - RENDIMIENTO[j]) / (RENDIMIENTO[i] - RENDIMIENTO[j])
        if 0 <= parte_i <= 1:
            pesos = np.zeros(N); pesos[i] = parte_i; pesos[j] = 1 - parte_i
            mayor = max(mayor, riesgo_de(pesos))
    return mayor

def esta_en_region(pesos):
    """Un reparto está dentro de la región si su riesgo cae entre el borde izquierdo
    (menor riesgo posible) y el derecho (mayor riesgo posible) para su rendimiento."""
    ren, ries = rendimiento_de(pesos), riesgo_de(pesos)
    borde_izq = riesgo_de(repartir_con_menor_riesgo(ren))
    borde_der = mayor_riesgo_posible(ren)
    return borde_izq - 1e-9 <= ries <= borde_der + 1e-9

# ---------------- PASO 3: LOS TRES PORTAFOLIOS QUE COMPARAMOS ---------------
pesos_gmvp = repartir_con_menor_riesgo()                 # el de menor riesgo
pesos_igual = np.ones(N) / N                              # 25% a cada uno

# Tangente: el que da más rendimiento por cada unidad de riesgo (ratio de Sharpe)
def menos_sharpe(w): return -(rendimiento_de(w) - TASA_LIBRE_RIESGO) / riesgo_de(w)
pesos_tangente = minimize(menos_sharpe, np.ones(N) / N, method="SLSQP", bounds=[(0, 1)] * N,
                          constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}]).x

portafolios = {"GMVP": pesos_gmvp, "Reparto parejo": pesos_igual, "Tangente": pesos_tangente}

# ------------- PASO 4: DIBUJAR LA REGIÓN Y LA FRONTERA ---------------------
# Recorremos rendimientos desde el menor hasta el mayor y, para cada uno, calculamos
# el menor y el mayor riesgo posible. Entre esas dos líneas está la región factible.
niveles = np.linspace(RENDIMIENTO.min(), RENDIMIENTO.max(), 300)
pesos_borde = [repartir_con_menor_riesgo(r) for r in niveles]
borde_izq = np.array([riesgo_de(w) for w in pesos_borde])
borde_der = np.array([mayor_riesgo_posible(r) for r in niveles])
es_eficiente = niveles >= rendimiento_de(pesos_gmvp)      # por encima del GMVP = eficiente

plt.figure(figsize=(10, 6.5))
plt.fill_betweenx(niveles, borde_izq, borde_der, color="#9ecae1", alpha=0.55, label="Región factible")
plt.plot(borde_izq[~es_eficiente], niveles[~es_eficiente], "--", color="gray", label="Parte ineficiente")
plt.plot(borde_izq[es_eficiente], niveles[es_eficiente], color="crimson", lw=2.5, label="Frontera eficiente")
plt.plot(borde_der, niveles, color="#3182bd", lw=1.2)
for i, nombre in enumerate(ACTIVOS):                      # cada activo solo, como punto negro
    plt.scatter(np.sqrt(COV[i, i]), RENDIMIENTO[i], color="black", zorder=5)
    plt.annotate(nombre, (np.sqrt(COV[i, i]), RENDIMIENTO[i]), xytext=(6, 4), textcoords="offset points")
estilo = {"GMVP": ("*", "gold", 320), "Reparto parejo": ("D", "purple", 70), "Tangente": ("^", "darkgreen", 110)}
for nombre, w in portafolios.items():
    m, c, s = estilo[nombre]
    plt.scatter(riesgo_de(w), rendimiento_de(w), marker=m, color=c, s=s, edgecolor="black", zorder=6, label=nombre)
# Línea del mercado de capitales (CAL): combinar el activo sin riesgo con el tangente
sharpe_t = -menos_sharpe(pesos_tangente)
x = np.linspace(0, borde_der.max() * 1.05, 50)
plt.plot(x, TASA_LIBRE_RIESGO + sharpe_t * x, "-.", color="darkgreen", label="CAL (con activo sin riesgo)")
plt.xlabel("Riesgo (volatilidad anual)"); plt.ylabel("Rendimiento esperado anual")
plt.title("Región factible y frontera eficiente"); plt.grid(alpha=0.3)
plt.legend(loc="lower right", fontsize=8); plt.xlim(0); plt.tight_layout()
CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas")
os.makedirs(CARPETA, exist_ok=True)
plt.savefig(os.path.join(CARPETA, "region_factible.png"), dpi=170)

# ---------------------- PASO 5: RESULTADOS EN PALABRAS ----------------------
print("RESULTADOS\n" + "-" * 60)
tabla = []
for nombre, w in portafolios.items():
    ren, ries = rendimiento_de(w), riesgo_de(w)
    dentro = esta_en_region(w)
    tabla.append([nombre, ren, ries, (ren - TASA_LIBRE_RIESGO) / ries, dentro] + list(w))
    print(f"{nombre:15s} rendimiento {ren:6.2%} | riesgo {ries:6.2%} | "
          f"{'DENTRO' if dentro else 'FUERA'} de la región | pesos: " +
          ", ".join(f"{a} {p:.1%}" for a, p in zip(ACTIVOS, w)))
tabla = pd.DataFrame(tabla, columns=["portafolio", "rendimiento", "riesgo", "sharpe", "en_region"] + ACTIVOS)
tabla.to_csv(os.path.join(CARPETA, "portafolios.csv"), index=False)

riesgo_sin_diversificar = pesos_gmvp @ np.sqrt(np.diag(COV))
print(f"\nDiversificación: si los activos se movieran idénticos, el GMVP tendría riesgo de "
      f"{riesgo_sin_diversificar:.2%}; en realidad tiene {riesgo_de(pesos_gmvp):.2%}. "
      f"Esa diferencia es lo que se ganó por mezclar activos distintos.")
print("\nCONCLUSIÓN: el GMVP SÍ está dentro de la región factible, justo en su punta izquierda: "
      "es el reparto de menor riesgo posible. Estar dentro solo significa que se puede armar; "
      "el 'reparto parejo' también está dentro, pero queda por debajo de la frontera "
      "(hay otro con el mismo riesgo y más rendimiento).")
print("\nGráfica y tabla guardadas en:", CARPETA)

# ----------------------- PASO 6 (OPCIONAL): MYSQL ---------------------------
# Guarda activos, parámetros y portafolios en una base de datos. Para activarlo:
#   1) pip install mysql-connector-python   2) USAR_MYSQL = True   3) pon tu password arriba.
if USAR_MYSQL:
    try:
        import mysql.connector
        con = mysql.connector.connect(**MYSQL); cur = con.cursor()
        cur.execute("DROP DATABASE IF EXISTS markowitz")      # ojo: borra y recrea esta base
        cur.execute("CREATE DATABASE markowitz"); cur.execute("USE markowitz")
        cur.execute("CREATE TABLE activos (id_activo INT PRIMARY KEY, ticker VARCHAR(15), "
                    "rendimiento_anual DOUBLE)")
        cur.execute("CREATE TABLE covarianzas (id_i INT, id_j INT, cov_anual DOUBLE, PRIMARY KEY (id_i, id_j))")
        cur.execute("CREATE TABLE portafolios (id_portafolio INT PRIMARY KEY, nombre VARCHAR(40), "
                    "rendimiento DOUBLE, riesgo DOUBLE, sharpe DOUBLE, en_region TINYINT(1))")
        cur.execute("CREATE TABLE pesos (id_portafolio INT, id_activo INT, peso DOUBLE, "
                    "PRIMARY KEY (id_portafolio, id_activo))")
        cur.executemany("INSERT INTO activos VALUES (%s,%s,%s)",
                        [(i + 1, ACTIVOS[i], float(RENDIMIENTO[i])) for i in range(N)])
        cur.executemany("INSERT INTO covarianzas VALUES (%s,%s,%s)",
                        [(i + 1, j + 1, float(COV[i, j])) for i in range(N) for j in range(N)])
        for k, fila in tabla.iterrows():
            cur.execute("INSERT INTO portafolios VALUES (%s,%s,%s,%s,%s,%s)",
                        (k + 1, fila.portafolio, float(fila.rendimiento), float(fila.riesgo),
                         float(fila.sharpe), int(fila.en_region)))
            cur.executemany("INSERT INTO pesos VALUES (%s,%s,%s)",
                            [(k + 1, i + 1, float(fila[a])) for i, a in enumerate(ACTIVOS)])
        con.commit()
        # Consulta de verificación: los pesos de cada portafolio deben sumar 1
        cur.execute("SELECT p.nombre, ROUND(SUM(w.peso),6) FROM portafolios p "
                    "JOIN pesos w USING (id_portafolio) GROUP BY p.nombre")
        print("\nMySQL cargado. Pesos suman:", cur.fetchall())
        con.close()
    except Exception as e:
        print("\n[MySQL] No se pudo cargar. Revisa usuario, password y que el servidor esté prendido:", e)
