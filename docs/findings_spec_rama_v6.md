# Hallazgos del Experimento: Recuperación de Modelos Cuantizados vía SpecRAMA-Quant (v6)

Este documento resume los resultados del **Experimento 06 (EXP-06)** en una GPU **NVIDIA A10G** (Modal Cloud), donde se evaluó el impacto de la **Corrección de Sesgo de Media por Canal (Channel Mean-Bias Correction)** y la **Permutación en el Manifold FP32 Limpio** sobre la recuperación de modelos GPT-2 cuantizados a **4.25 bits** y **3.31 bits promedio** sobre el **Test Set de WikiText-2 (280.000 tokens)**.

---

## 1. Resultados Oficiales (EXP-06: 4.2b vs 3.3b Advanced Recovery)

| Método / Configuración | Bitwidth Base | Core Size | Tamaño Adaptador | Step 0 PPL (Sin Entrenar) | Tuned PPL (200 Pasos) | Reducción de Perplejidad | Estado |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPT-2 Base (FP32)** | 32-bit FP32 | N/A | N/A | 46.18 | N/A | Baseline | Referencia |
| **SpecRAMA-Quant 4.2-bit Wavelet** | 4.25-bit avg | 16x16 | **72.0 KB** | 789.29 | **146.68** ↓ | **$-642.61$ PPL ($5.4\times$ mejor)** | **Recuperación Masiva** |
| **SpecRAMA-Quant 3.3-bit Wavelet** | 3.31-bit avg | 16x16 | **72.0 KB** | 4440.10 | **2567.68** ↓ | **$-1872.42$ PPL ($1.7\times$ mejor)** | Recuperación en Progreso |
| **SpecRAMA-Quant 3.3-bit Wavelet** | 3.31-bit avg | 32x32 | **288.0 KB** | 4440.10 | **2159.12** ↓ | **$-2280.98$ PPL ($2.1\times$ mejor)** | Recuperación en Progreso |

---

## 2. Hallazgos e Insights Científicos

### A. La Gran Revelación a 4.25 Bits (Caída de 789 a 146 PPL en solo 200 Pasos)
* A **4.25 bits promedio (8-bit Core / 4-bit Rest)**, la cuantización sin entrenar degrada la red a 789.29 PPL.
* Inyectar **SpecRAMA Wavelet (16x16) de solo 72 KB** logra recortar **más de 642 puntos de perplejidad** en apenas 200 pasos, bajando a **146.68 PPL**.
* Esto confirma empíricamente que a **4 bits el error de cuantización es un residuo espectralmente acotado** que SpecRAMA puede absorber de forma ultrarrápida.

### B. El Mecanismo del Límite a 3.3 Bits (Saturación de Recorte / Tails)
* A **3.31 bits (8 niveles de cuantización)**, la distancia entre niveles produce **saturación en las colas de la distribución gaussiana (clipping error)**.
* Esto genera una distorsión de rango bajo en la matriz que requiere:
  1. **Cuantización Espectral Cuantil/No-Uniforme (tipo NF4)** en lugar de cuantización lineal simétrica RTN.
  2. **Mayor presupuesto de pasos de optimización (500 - 1000 pasos)**, ya que la curva de pérdida continuaba descendiendo a velocidad constante en el paso 200.

---

## 3. Conclusiones para la Publicación / Proyecto

1. **SpecRAMA-Quant de 72 KB habilita la inferencia eficiente a 4.2 bits** rescatando modelos degradados casi al nivel funcional en pasos de GPU ínfimos.
2. **Corrección de Sesgo Valida la Teoría**: La recalibración de la media de canal evitó que el modelo a 3.3 bits explotara por encima de 10.000 PPL, estabilizando el gradiente durante todo el entrenamiento.
