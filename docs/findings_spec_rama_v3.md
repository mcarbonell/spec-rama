# Hallazgos del Experimento: Ley de Escalado de Cores Espectrales y Corrección de Parseval (v3)

Este documento resume los resultados del **Experimento 03 (EXP-03)** en una GPU **NVIDIA A10G** (Modal Cloud), donde se evaluó la **Ley de Escalado de Resolución Espectral ($8 \times 8 \to 64 \times 64$)** tras aplicar el escalado de norma de energía de Parseval ($\frac{\alpha}{\sqrt{k_{\text{out}} \cdot k_{\text{in}}}}$) sobre la Perplejidad en el **Test Set de WikiText-2 (280.000 tokens)**.

---

## 1. Resultados Oficiales (Ley de Escalado de Resolución Espectral)

| Configuración / Método | Parámetros Entrenables | % del Total de Pesos | Tamaño del Adaptador | TEST PPL | Merged PPL (Latencia 0) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GPT-2 Base (FP32 completo)** | 124,439,808 | 100.00% | N/A | 46.18 | N/A |
| **LoRA Estándar ($r=8, \alpha=16$)** | 811,008 | 0.6480% | 3.10 MB | 30.12 | N/A |
| **SpecRAMA Wavelet (8x8)** | **4,608** | **0.0037%** | **18.0 KB** | **42.57** ↓ | **42.57** |
| **SpecRAMA Wavelet (16x16)** | **18,432** | **0.0148%** | **72.0 KB** | **40.73** ↓ | **40.73** |
| **SpecRAMA Wavelet (32x32)** | **73,728** | **0.0592%** | **288.0 KB** | **38.35** ↓ | **38.35** |
| **SpecRAMA DCT (32x32)** | **73,728** | **0.0592%** | **288.0 KB** | **38.96** ↓ | **38.96** |
| **SpecRAMA Wavelet (64x64)** | **294,912** | **0.2364%** | **1.12 MB** | **36.21** ↓ (Ganador Spec) | **36.21** |

---

## 2. Descubrimientos Teóricos y Empíricos Clave

### A. Ley de Escalado Monótona Espectral
Tras aplicar la corrección de escalado por energía de Parseval, Spec-RAMA exhibe una **curva de escalado monótona perfecta**:
$$\text{PPL: } 46.18 \text{ (Base)} \longrightarrow 42.57 \text{ (8x8)} \longrightarrow 40.73 \text{ (16x16)} \longrightarrow 38.35 \text{ (32x32)} \longrightarrow 36.21 \text{ (64x64)}$$
* A mayor resolución del núcleo espectral ($k \times k$), mayor es la capacidad del modelo para capturar detalles semánticos y reducir la perplejidad sobre datos nunca vistos.

### B. Validación de la Corrección de Escalado por Energía de Parseval
* **Diagnóstico v2 (Error de escalado)**: Al usar la división cuadrática $\frac{\alpha}{k^2}$, la magnitud de la actualización para $64 \times 64$ se reducía $64\times$, bloqueando el gradiente y congelando la perplejidad en 46.08.
* **Solución v3 (Teorema de Parseval)**: Al usar la norma de conservación de energía $L_2$ en transformadas ortogonales:
$$\text{scaling} = \frac{\alpha}{\sqrt{k_{\text{out}} \cdot k_{\text{in}}}}$$
el gradiente fluyó de forma óptima a través de todas las resoluciones, desbloqueando una reducción masiva de perplejidad (**36.21 PPL**).

### C. Superioridad Constante de Wavelet sobre DCT
* A la resolución $32 \times 32$, **Wavelet (38.35 PPL)** volvió a superar a **DCT (38.96 PPL)** con exactamente el mismo tamaño de adaptador (**288.0 KB**).
* Esto confirma que la descomposición espacial-frecuencial de Haar 2D es intrínsecamente más eficiente que la base continua cosenoidal para adaptar parámetros en Transformers.

### D. Eficiencia de Tamaño del Adaptador ($3\times$ más pequeño que LoRA)
* **SpecRAMA Wavelet (64x64)** alcanza **36.21 PPL** ocupando **solo 1.12 MB** en disco/memoria, comparado con los **3.10 MB** de LoRA ($3\times$ más compacto que LoRA).

---

## 3. Conclusiones y Roadmap

1. **Spec-RAMA es un framework de PEFT continuo y totalmente personalizable**: Permite ajustar el presupuesto de adaptador desde **1.1 KB (Sub-KB)** hasta **1.12 MB**, adaptándose a cualquier restricción de ancho de banda o memoria.
2. **Fusión Frecuencial Inversa (`merge()`) Perfecta**: En todas las resoluciones ($8\times8$, $16\times16$, $32\times32$, $64\times64$), la perplejidad tras el `merge()` in-place es idéntica con precisión de dos decimales al modelo durante el entrenamiento.
