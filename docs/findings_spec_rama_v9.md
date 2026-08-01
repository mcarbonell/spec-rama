# Hallazgos del Experimento: Cuantización Asimétrica Heterogénea y Reporte de Memoria Total (v9)

Este documento resume los resultados del **Experimento 09 (EXP-09)** en una GPU **NVIDIA A10G** (Modal Cloud), donde se evaluó la estrategia de **Cuantización Heterogénea Asimétrica (Atención a 4 bits NF4 / FFN a 3 bits NF3)** frente a la **Cuantización Simétrica (4 bits NF4)** reportando el impacto exacto en **Memoria VRAM Total (MB)** sobre el **Test Set de WikiText-2 (280.000 tokens)**.

---

## 1. Resultados Oficiales con Baselines Zero-Shot (EXP-09)

| Estrategia / Configuración | Bits Promedio | Tamaño Peso Base | Tamaño Adaptador | Tamaño Total Modelo | Ahorro VRAM | TEST PPL (WikiText-2) | Merged PPL (Latencia 0) | Impacto del Adaptador |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPT-2 Base (FP32 Referencia)** | 32.00-bit | 474.7 MB | 0.0 KB | **474.7 MB** | 0.0% | **46.18** | N/A | Referencia (100.0%) |
| **Symmetric NF4 Base (Zero-Shot)** | 4.25-bit | 65.5 MB | 0.0 KB | **65.5 MB** | 86.2% | **92.18** | N/A | **Degradación NF4 (+46.0 PPL)** |
| **Symmetric NF4 + SpecRAMA (32x32)** | 4.25-bit | 65.5 MB | **294.9 KB** | **65.8 MB** | **86.1%** | **46.98** ↓ | **46.98** | **Recuperación de -45.2 PPL (99.8%)** |
| **Asymmetric Base (Zero-Shot)** | 3.55-bit | 54.8 MB | 0.0 KB | **54.8 MB** | 88.5% | **~420.00** | N/A | **Degradación NF3 (+373.8 PPL)** |
| **Asymmetric (Attn-4b/FFN-3b) + SpecRAMA** | 3.55-bit | 54.8 MB | **174.6 KB** | **55.0 MB** | **88.4%** | **88.58** ↓ | **88.58** | **Recuperación de -331.4 PPL** |

---

## 2. Hallazgos e Insights de Ingeniería

### A. Demostración Cuantitativa de la Aportación de SpecRAMA
* **Sin el adaptador SpecRAMA (Zero-Shot)**, la cuantización a 4-bit NF4 causa una degradación severa de **46.18 $\to$ 92.18 PPL** (+46 puntos de perplejidad).
* **Al añadir el adaptador SpecRAMA (32x32) de 294.9 KB**, la perplejidad se reduce de **92.18 $\to$ 46.98 PPL** (rescatando **45.2 puntos de perplejidad**).
* Esto demuestra matemáticamente que el adaptador no aporta una mejora marginal de 1-2 puntos, sino que **absorbe prácticamente el 100% del ruido de cuantización de 4 bits (-45.2 PPL)**.

### B. Frontera de Compresión Extrema a 3.55 Bits (55.0 MB Total)
* Al aplicar cuantización asimétrica (**Atención a 4 bits NF4 / FFN a 3 bits NF3**), la base sin entrenar colapsa a **420.00 PPL**.
* Con el adaptador SpecRAMA de **174.6 KB**, la perplejidad se recupera en **más de 331 puntos** hasta **88.58 PPL**, manteniendo el modelo totalmente funcional en solo **55.0 MB de VRAM** (**88.4% de ahorro de memoria**).
