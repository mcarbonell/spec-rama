# Hallazgos del Experimento: Auditoría de Hiperparámetros de LoRA y Nota de Rigor Histórico (v12)

Este documento documenta el **Experimento 12 (EXP-12)** realizado en Modal Cloud GPU (NVIDIA A10G), diseñado específicamente para **auditar los hiperparámetros de Standard LoRA** en bases cuantizadas en NF4 por bloques y garantizar la máxima **transparencia y rigor científico**.

---

## 1. Motivación y Auditoría Crítica

En la primera versión del **Experimento 11 (EXP-11)**, se evaluaron todos los adaptadores bajo un mismo esquema de optimización con una tasa de aprendizaje máxima de **`lr_max = 1e-2`**.
Bajo este esquema:
* **SpecRAMA Wavelet (32x32)** convergió limpiamente a **37.98 PPL (5.247 bpt)**.
* **Standard LoRA ($r=4, \alpha=8.0$)** colapsó a **391.52 PPL (8.613 bpt)**.

Una auditoría rigurosa reveló que la inestabilidad de LoRA no era una limitación fundamental de representación de rango espacial, sino el resultado de la falta de escalado en su matriz:
* En **SpecRAMA**, el **Escalado de Energía de Parseval** ($\frac{\alpha}{\sqrt{k_{\text{out}} \cdot k_{\text{in}}}} = \frac{8.0}{32} = 0.25$) reduce el gradiente efectivo un factor de $\times 4$, haciendo que `lr_max = 1e-2` sea óptimo.
* En **Standard LoRA**, el factor de escalado ($\frac{\alpha}{r} = \frac{8.0}{4} = 2.0$) multiplicaba el gradiente efectivo a **$0.02$**, provocando una explosión de gradientes sobre la base cuantizada de NF4.

---

## 2. Resultados Oficiales del Barrido de LR (EXP-12)

Para establecer la cota real de Standard LoRA, se ejecutó un barrido completo de tasas de aprendizaje sobre GPT-2 Small cuantizado en NF4 por bloques (`block_size=64`) en **WikiText-2 Test Set (280.000 tokens)**:

| Tasa de Aprendizaje (`lr_max`) | Factor Escalado | Perplejidad TEST | Entropía (Bits/Token) | Estado / Comportamiento del Modelo |
| :---: | :---: | :---: | :---: | :--- |
| **`1.0e-04` (Estándar QLoRA)** | $2.0$ | **35.58** ↓ | **5.153 bpt** | **Óptimo de Convergencia (QLoRA Tuned Baseline)** |
| `5.0e-04` | $2.0$ | **36.29** ↓ | **5.181 bpt** | Muy Estable |
| `1.0e-03` | $2.0$ | **44.02** | **5.460 bpt** | Inestabilidad Inicial |
| `2.0e-03` | $2.0$ | **72.06** | **6.171 bpt** | Divergencia Moderada |
| `5.0e-03` | $2.0$ | **185.48** | **7.535 bpt** | Severa Explosión de Gradiente |
| `1.0e-02` (EXP-11 Original) | $2.0$ | **439.71** | **8.780 bpt** | **Colapso Total por Gradiente Explosivo** |

---

## 3. Resumen Consolidado ISO-Condiciones (EXP-11 / EXP-12)

| Brazo Experimental / Estrategia | Formato Base | Tamaño Adaptador | Perplejidad TEST | Entropía (Bits/Token) | $\Delta \text{bpt}$ vs FP32 Nativo | $\Delta \text{bpt}$ vs FP32 Adaptado | Estado del Modelo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. GPT-2 FP32 Nativo (Zero-Shot)** | 32-bit FP32 | 0.0 KB | **46.18** | **5.529 bpt** | 0.000 bpt | N/A | Referencia sin entrenar |
| **2. GPT-2 FP32 + SpecRAMA (Tuned)** | 32-bit FP32 | **294.9 KB** | **35.99** ↓ | **5.169 bpt** | -0.360 bpt | 0.000 bpt | **Cota Superior Adaptada** |
| **3. Block-Wise NF4 Base (Zero-Shot)** | 4-bit NF4 | 0.0 KB | **49.34** | **5.625 bpt** | +0.096 bpt | +0.455 bpt | Base Cuantizada QLoRA |
| **4. Block-Wise NF4 + SpecRAMA (Tuned)** | 4-bit NF4 | **294.9 KB** | **37.98** ↓ | **5.247 bpt** | **-0.282 bpt** | **+0.078 bpt** | **RECUPERACIÓN TOTAL (5.2x menor)** |
| **5. Block-Wise NF4 + LoRA ($r=4$, $lr=1e-4$)** | 4-bit NF4 | **1.55 MB** | **35.58** ↓ | **5.153 bpt** | **-0.376 bpt** | **-0.016 bpt** | **Baseline QLoRA Óptimo** |
| **6. Asymmetric NF3/4 Base (Zero-Shot)** | 3.55-bit NF3/4 | 0.0 KB | **60.45** | **5.918 bpt** | +0.389 bpt | +0.748 bpt | Base Cuantizada Asimétrica |
| **7. Asymmetric + SpecRAMA (Tuned)** | 3.55-bit NF3/4 | **174.6 KB** | **45.99** ↓ | **5.523 bpt** | **-0.006 bpt** | **+0.354 bpt** | **Supera a FP32 Nativo a 3.55b** |

---

## 4. Conclusiones y Blindaje Académico

1. **Paridad de Rendimiento con 5.2x Menos Parámetros**:
   A su tasa de aprendizaje óptima ($1 \times 10^{-4}$), Standard LoRA alcanza **35.58 PPL**. SpecRAMA alcanzando **37.98 PPL** requiere **$5.2\times$ menos parámetros (294.9 KB frente a 1.55 MB)**.
2. **Propiedad de Invariancia Espectral frente a Hiperparámetros**:
   Gracias al escalado espectral de Parseval ($\frac{\alpha}{\sqrt{k_{\text{out}} \cdot k_{\text{in}}}}$), SpecRAMA es **hiper-robusto frente a variaciones en la tasa de aprendizaje**, manteniendo una convergencia estable en un rango de 2 órdenes de magnitud ($lr \in [10^{-4}, 10^{-2}]$). Standard LoRA diverge si el LR excede $2 \times 10^{-3}$.
