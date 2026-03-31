# Clasificación de Modo de Transporte desde GPS con CNN

Clasificación automática de modo de transporte (caminar, bicicleta, bus, auto, tren) a partir de trayectorias GPS crudas usando una Red Neuronal Convolucional en PyTorch.

**Test Accuracy: 77.5%** en 25,449 muestras — competitivo con el paper seminal de Zheng et al. (2008) que reportó 76.2% en el mismo dataset.

---

## Problema

Las ciudades necesitan saber cómo se mueven sus ciudadanos — ¿caminan, usan bicicleta, toman bus, manejan auto o usan tren? Las encuestas tradicionales cuestan millones y tardan meses. Este modelo lo resuelve automáticamente a partir de datos GPS de celulares.

## Enfoque

Puntos GPS crudos `(lat, lon, timestamp)` se transforman en features cinemáticos y se clasifican con una CNN:

```
Trayectoria GPS → Extracción de features → Ventanas deslizantes → CNN → Modo de transporte
                   (velocidad, aceleración,    (matriz 60 × 5)
                    bearing, jerk)
```

1. **Feature Engineering**: calcular distancia, velocidad, aceleración, cambio de dirección y jerk entre puntos GPS consecutivos
2. **Ventanas**: dividir trayectorias en ventanas fijas de 60 timesteps con 50% de solapamiento
3. **Clasificación CNN**: red Conv2d de 3 bloques con BatchNorm, MaxPool y Global Average Pooling

## Resultados

| Modo | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Walk | 78.2% | 81.4% | 79.8% |
| Bike | 75.7% | 83.8% | 79.6% |
| Bus | 83.0% | 64.2% | 72.4% |
| Car | 65.1% | 75.7% | 70.0% |
| Train | 84.0% | 84.2% | **84.1%** |
| **General** | | | **77.5%** |

**Mejor clase**: Train (84.1% F1) — velocidad alta + dirección estable es inconfundible.  
**Clase más difícil**: Bus vs Car — velocidades similares en tráfico urbano. Papers que alcanzan 90%+ agregan datos de acelerómetro o proximidad a estaciones de transporte.

### Contexto en la Literatura

| Paper | Accuracy | Método |
|-------|----------|--------|
| Zheng et al. (2008) | 76.2% | Decision Tree + CRF |
| Dabiri & Heaslip (2018) | 84.8% | CNN profunda |
| Ensemble CNN (2019) | 91.8% | CNN + Random Forest |
| **Este proyecto** | **77.5%** | **CNN (PyTorch)** |

## Dataset

**[GeoLife GPS Trajectories](https://www.microsoft.com/en-us/download/details.aspx?id=52367)** — Microsoft Research Asia

- 182 usuarios, 17,621 trayectorias, 24M+ puntos GPS (Beijing, 2007–2012)
- 69 usuarios con etiquetas de modo de transporte anotadas manualmente (14,692 segmentos)
- Después del ventaneo: **169,655 muestras** (60 timesteps × 5 features cada una)

### Configuración

1. Descargar GeoLife del link de arriba
2. Extraer y colocar la carpeta `Data/` en el mismo directorio que el notebook
3. Ejecutar todas las celdas en orden

## Arquitectura — TransportNet

```
Input: (batch, 1, 60, 5)

Bloque 1: Conv2d(1→32, 3×1) + BN + ReLU + Conv2d(32→32) + BN + ReLU + MaxPool(2×1)
Bloque 2: Conv2d(32→64, 3×3) + BN + ReLU + Conv2d(64→64) + BN + ReLU + MaxPool(2×1)
Bloque 3: Conv2d(64→128, 3×3) + BN + ReLU + MaxPool(2×1)

Global Average Pooling → Dropout(0.4) → Linear(128→64) → ReLU → Linear(64→5)

Parámetros totales: 141,733
```

Entrenamiento: AdamW optimizer, CrossEntropyLoss con class weights, ReduceLROnPlateau scheduler, early stopping.

## Stack Tecnológico

`Python` · `PyTorch` · `CUDA` · `NumPy` · `Pandas` · `Matplotlib` · `Scikit-learn`

## Referencias

- Zheng, Y. et al. (2008). *Understanding Mobility Based on GPS Data*. UbiComp 2008.
- Dabiri, S. & Heaslip, K. (2018). *Inferring transportation modes from GPS trajectories using CNN*. Transportation Research Part C, 86, 360-371.
- Lopes et al. (2024). *A deep learning approach for transportation mode identification*. Int. J. Data Science and Analytics.

## Autor

**Mario Carvajal** — Economista aplicando Deep Learning a problemas de movilidad urbana.

---

*Este proyecto es parte de una serie de portafolio aplicando Machine Learning a problemas económicos reales.*
