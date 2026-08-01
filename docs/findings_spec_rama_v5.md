# Hallazgos del Experimento: Recuperación de Cuantización Espectral a 3.3 Bits vía Spec-RAMA (v5)

Este documento resume los resultados del **Experimento 05 (EXP-05)** en una GPU **NVIDIA A10G** (Modal Cloud), donde se evaluó la capacidad de **Spec-RAMA** para recuperar la precisión de un modelo GPT-2 Small severamente colapsado por **Cuantización Espectral Jerárquica a 3.3 bits promedio (8-bit Core / 3-bit Rest)** sobre el **Test Set de WikiText-2 (280.000 tokens)**.

---

## 1. Resultados Oficiales (EXP-05: 3.3-Bit Recovery)

| Método / Configuración | Bitwidth Base | Parámetros Entrenables | Tamaño Adaptador | Perplejidad en TEST | Estado del Modelo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GPT-2 Base (FP32)** | 32-bit FP32 | 124,439,808 | N/A | **46.18** | Referencia sin comprimir |
| **SpecQuant 3.3-bit Zero-Shot** | 3.31-bit avg | 0 (Quantized) | 0 KB | **4498.40** | **Colapsado Estático** |
| **SpecRAMA-Quant DCT (8x8)** | 3.31-bit avg | 4,608 | **18.0 KB** | **2835.89** ↓ | **Recuperación Parcial ($1.6\times$)** |
| **SpecRAMA-Quant Wavelet (8x8)** | 3.31-bit avg | 4,608 | **18.0 KB** | **2976.94** ↓ | **Recuperación Parcial ($1.5\times$)** |
| **SpecRAMA-Quant Wavelet (16x16)** | 3.31-bit avg | 18,432 | **72.0 KB** | **2491.67** ↓ (Ganador v5) | **Recuperación Parcial ($1.8\times$)** |

---

## 2. Hallazgos e Insights Científicos

### A. Capacidad de Rescate Espectral (Reducción de PPL de 4498 a 2491)
* El modelo sin fine-tuning a 3.31 bits promedio (8-bit Core / 3-bit Rest) sufre un **colapso dinámico catastrófico (4498.40 PPL)** debido a la severa restricción de rango de los 8 niveles de cuantización.
* Añadir únicamente un adaptador espectral **SpecRAMA Wavelet (16x16) de 72 KB** redujo la perplejidad de **4498 a 2491 PPL** tras solo 200 pasos de entrenamiento en GPU.
* Esto demuestra que los gradientes frecuenciales **SÍ tienen capacidad para amortiguar el ruido de redondeo a 3 bits**.

### B. Diagnóstico de la Barrera Residual de 2491 PPL
Al analizar el comportamiento de EXP-05, se identificaron **dos cuellos de botella estructurales**:
1. **Sesgo/Offset por Canal (DC Offset)**: La cuantización uniforme a 3 bits ($[-4, \dots, 3]$) introduce un desplazamiento medio diferente de cero en las columnas de proyección. Multiplicar la matriz cuantizada por el adaptador mantenía anclado este sesgo.
2. **Permutación Deformada Post-Cuantización**: En EXP-05 las permutaciones $P_L, P_R$ se calculaban sobre los pesos ya ruidosos a 3 bits en lugar de la variedad suave original en FP32.

---

## 3. Transición hacia el Experimento 06 (`EXP-06`)

Para romper la barrera de los 2491 PPL y lograr una recuperación limpia, el Experimento 06 implementa:
* **Corrección de Sesgo de Media por Canal (`Channel Mean-Bias Correction`)**.
* **Cálculo de Permutaciones $P_L, P_R$ sobre los pesos limpios en FP32**.
* **Recuperación Residual Aditiva y Comparativa entre 4.2 bits y 3.3 bits**.
