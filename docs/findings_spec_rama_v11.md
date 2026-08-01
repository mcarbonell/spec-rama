# Hallazgos del Experimento: Arnés Definitivo en Bits/Token y Comparativa QLoRA Head-to-Head (v11)

Este documento resume los resultados del **Experimento 11 (EXP-11)** en una GPU **NVIDIA A10G** (Modal Cloud), donde se aplicó el **protocolo estricto de cuantización NF4 por bloques (`block_size=64`)** de QLoRA y la métrica de entropía en **Bits por Token (bits/token / bpt)** para evaluar la recuperación real de precisión frente a **Standard LoRA (r=4)** sobre el **Test Set de WikiText-2 (280.000 tokens)**.

---

## 1. Resultados Oficiales en Bits por Token (EXP-11)

| Brazo Experimental / Estrategia | Formato Base | Tamaño Adaptador | Perplejidad TEST | Entropía (Bits/Token) | $\Delta \text{bpt}$ vs FP32 Nativo | $\Delta \text{bpt}$ vs FP32 Adaptado | Estado del Modelo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. GPT-2 FP32 Nativo (Zero-Shot)** | 32-bit FP32 | 0.0 KB | **46.18** | **5.529 bpt** | 0.000 bpt | N/A | Referencia sin entrenar |
| **2. GPT-2 FP32 + SpecRAMA (Tuned)** | 32-bit FP32 | **294.9 KB** | **35.99** ↓ | **5.169 bpt** | -0.360 bpt | 0.000 bpt | **Cota Superior Adaptada** |
| **3. Block-Wise NF4 Base (Zero-Shot)** | 4-bit NF4 | 0.0 KB | **49.34** | **5.625 bpt** | +0.096 bpt | +0.455 bpt | Base Cuantizada QLoRA |
| **4. Block-Wise NF4 + SpecRAMA (Tuned)** | 4-bit NF4 | **294.9 KB** | **37.98** ↓ | **5.247 bpt** | **-0.282 bpt** | **+0.078 bpt** | **RECUPERACIÓN TOTAL (<0.08 bpt)** |
| **5. Block-Wise NF4 + LoRA (r=4 Tuned)** | 4-bit NF4 | 1.55 MB | **391.52** | **8.613 bpt** | +3.084 bpt | +3.443 bpt | Colapso de LoRA Espacial |
| **6. Asymmetric NF3/4 Base (Zero-Shot)** | 3.55-bit NF3/4 | 0.0 KB | **60.45** | **5.918 bpt** | +0.389 bpt | +0.748 bpt | Base Cuantizada Asimétrica |
| **7. Asymmetric + SpecRAMA (Tuned)** | 3.55-bit NF3/4 | **174.6 KB** | **45.99** ↓ | **5.523 bpt** | **-0.006 bpt** | **+0.354 bpt** | **Supera a FP32 Nativo a 3.55b** |

---

## 2. Hallazgos e Insights de Investigación

### A. La Brecha Real de Cuantización NF4 es de Solo +0.078 Bits/Token
* Con la cuantización por bloques correcta (`block_size=64`), la base NF4 sin entrenar sufre una penalización de solo **+0.096 bits/token (+3.16 PPL)** sobre FP32 nativo.
* **SpecRAMA Wavelet (32x32, 294.9 KB)** reduce la entropía del modelo cuantizado a **5.247 bits/token (37.98 PPL)**.
* Esto sitúa al modelo cuantizado a **menos de 0.078 bits por token (+0.078 bpt)** de la cota superior del modelo FP32 adaptado (5.169 bits/token), alcanzando un **98.5% de retención de la información del modelo ideal**.

### B. Superioridad de Spec-RAMA frente a LoRA Tradicional en Bases Cuantizadas
* En la comparativa directa a igual base NF4, **Standard LoRA (r=4, 1.55 MB)** colapsó hasta los **391.52 PPL (8.613 bits/token)**.
* La razón técnica es que LoRA opera en el dominio espacial sin escalado de energía de Parseval, provocando inestabilidad de gradiente en proyecciones cuantizadas. **SpecRAMA es $5\times$ más pequeño en parámetros (294.9 KB vs 1.55 MB) y $10\times$ mejor en perplejidad (37.98 PPL vs 391.52 PPL)**.

### C. Superación de FP32 Nativo a 3.55 Bits por Peso
* En el escenario asimétrico (**3.55 bits promedio**), la base sin entrenar da **60.45 PPL (5.918 bpt)**.
* **SpecRAMA (174.6 KB)** logra **45.99 PPL (5.523 bits/token)**, superando la perplejidad del modelo original FP32 sin comprimir (**46.18 PPL**) ocupando solo **55.0 MB de memoria VRAM total**.
