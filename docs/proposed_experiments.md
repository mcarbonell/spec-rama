# Propuesta de Nuevos Experimentos: Spec-RAMA y la Hipótesis del Subespacio de Pesos Universal

**Fecha:** Septiembre 2026  
**Autor:** Mario Raúl Carbonell Martínez  
**Repositorio:** `spec-rama/`  
**Objetivo Metrológico:** Definir el plan experimental de segunda generación para `spec-rama`, conectando la arquitectura de núcleos espectrales con la *Hipótesis del Subespacio de Pesos Universal* (Kaushik, Chellappa, Yuille et al., 2025; arXiv:2512.05117), el *Principio de Representación Compacta* (Carbonell, 2026) y superando los límites identificados en la auditoría técnica de los experimentos EXP-11 y EXP-12.

---

## 1. Contexto Teórico y Estado Metrológico

### 1.1. Conexión con la Hipótesis del Subespacio de Pesos Universal (arXiv:2512.05117)
El trabajo reciente de Kaushik et al. (Johns Hopkins / Texas A&M, 2025) demuestra teórica y empíricamente que las redes neuronales sobreparametrizadas poseen una geometría oculta compartida: los desplazamientos óptimos de pesos entre múltiples tareas downstream residen en un **subespacio común de baja dimensión** gobernado por el operador de segundo momento de los pesos:
$$\Delta W \approx U_{\text{univ}} \, \Sigma \, V_{\text{univ}}^\top$$
Sin embargo, en dicho paper el subespacio universal se deriva *a posteriori* de forma costosa mediante HOSVD (Higher-Order Singular Value Decomposition) tras entrenar múltiples modelos completos densos.

**La tesis fundamental de `spec-rama` es que este subespacio no requiere entrenamiento previo de modelos masivos: puede sintetizarse de forma constructiva, determinista y analítica a partir del grafo geométrico de la matriz base $W_0$ mediante permutación TSP de canales y transformadas de frecuencia 2D (Wavelet / DCT)**:
$$\Delta W = P_L^\top \cdot \mathcal{T}^{-1}(S_{\text{core}}) \cdot P_R = U_{\text{spectral}} \cdot S_{\text{core}} \cdot V_{\text{spectral}}^\top$$
donde $U_{\text{spectral}} = P_L^\top B_{\text{out}}^\top$ y $V_{\text{spectral}} = B_{\text{in}} P_R$ actúan como las bases universales fijas, y el núcleo de adaptación $S_{\text{core}} \in \mathbb{R}^{k_{\text{out}} \times k_{\text{in}}}$ codifica la perturbación de la tarea.

### 1.2. Línea Base Post-Auditoría (EXP-11 y EXP-12)
Tras la auditoría técnica externa, se han establecido los estándares metrológicos estrictos:
1. **Resolución del artefacto de LoRA (391.52 PPL)**: Se demostró que dicho colapso en versiones preliminares no provenía de una incompatibilidad intrínseca entre matrices de bajo rango y pesos cuantizados, sino de optimizar $38.6\text{M}$ parámetros de la tabla de embeddings de vocabulario (`wte`) descongelados con AdamW a $lr=10^{-2}$. Con los embeddings debidamente congelados, LoRA ($r=4$, 589.824 parámetros / 2.25 MB) converge de forma estable a **28.78–33.38 PPL**.
2. **Frontera de Pareto Real**: Spec-RAMA simétrico $32\times 32$ alcanza **37.96 PPL (5.246 bpt)** con solo **98.304 parámetros (384.0 KB)**, logrando una **reducción de parámetros de $6.0\times$** frente a LoRA ($r=4$), recuperando el 98.5% de la información adaptada (+0.078 bpt frente al upper bound adaptado en FP32) y permitiendo fusión directa *in-place* sin sobrecoste de latencia.
3. **Régimen Asimétrico Extremo (3.55 bits)**: Spec-RAMA con núcleo de **204.0 KB** (52.224 parámetros) alcanza **45.95 PPL (5.522 bpt)**, **superando a la base FP32 nativa sin adaptar (46.18 PPL)** con una reducción de parámetros de **$11.3\times$** frente a LoRA.
4. **Clarificación del Protocolo**: Las evaluaciones de EXP-11 y EXP-12 se ejecutaron sobre 100 bloques (25.600 tokens) del split de test de WikiText-2 bajo cuantización simulada NF4 (`block_size=64`).

---

## 2. Programa Experimental Propuesto

```
+-----------------------------------------------------------------------------------+
|                            PROGRAMA EXPERIMENTAL SPEC-RAMA                        |
+-----------------------------------------------------------------------------------+
  |
  +---> EXP-18: Universalidad e Invarianza de la Permutación TSP (Base vs Tarea)
  |
  +---> EXP-19: Fusión Espectral Directa de Modelos (Arithmetic Task Merging)
  |
  +---> EXP-20: Subespacio Espectral Compartido Multi-Capa (SharedSpecRAMA Sub-10 KB)
  |
  +---> EXP-21: Adaptación Espectral en Visión (ViT en CIFAR con DirectML / CPU)
  |
  +---> EXP-22: Evaluación a Escala Completa (WikiText-2 280K tokens + Downstream GLUE)
```

---

### EXP-18: Universalidad e Invarianza de la Permutación TSP (Cross-Task Basis Invariance)

#### 1. Pregunta Científica
¿Constituyen las matrices de permutación $P_L, P_R$ obtenidas mediante TSP sobre el modelo base $W_0$ un sistema de coordenadas canónico y universal para la arquitectura, o requieren ser recalculadas específicamente para cada tarea downstream?

#### 2. Hipótesis Falsable
**H18:** El sistema de coordenadas espectral $P_L, P_R$ calculado únicamente sobre los pesos preentrenados $W_0$ de GPT-2 contiene la totalidad de las direcciones de máxima covarianza requeridas para tareas heterogéneas. Recomputar permutaciones adaptativas sobre gradientes de tarea o activaciones no produce una mejora de entropía superior a $0.05\text{ bpt}$ respecto al sistema fijo preentrenado.

#### 3. Diseño Experimental
* **Base Fija:** GPT-2 Small cuantizado en NF4 por bloques (`block_size=64`), embeddings estrictamente congelados.
* **Brazos de Comparación:**
  1. `Arm 1 (TSP Base Fija)`: $P_L, P_R$ computados sobre $W_0$ preentrenado (protocolo estándar Spec-RAMA).
  2. `Arm 2 (TSP Activación-Covarianza)`: $P_L, P_R$ computados a partir de la matriz de covarianza de activaciones en la tarea de destino.
  3. `Arm 3 (Sin Permutación / Identidad)`: $P_L = I, P_R = I$ (control negativo de ordenamiento).
  4. `Arm 4 (Permutación Aleatoria)`: $P_L, P_R$ aleatorias fijas (control negativo de geometría).
* **Dominios Heterogéneos:**
  - Dominio 1 (Lenguaje General): WikiText-2 Test.
  - Dominio 2 (Noticias / Estructura Sintáctica): Penn Treebank (PTB).
  - Dominio 3 (Código Estructurado): Python Code Token Slice (The Stack / CodeSearchNet).
* **Métricas:** Perplejidad (PPL), Bits por Token (bpt), Concentración de energía espectral ($\%$ de varianza en banda LL vs LH/HL/HH).

#### 4. Criterio de Falsación y Relevancia
Si `Arm 1` se mantiene a menos de $0.05\text{ bpt}$ de `Arm 2` en todos los dominios y supera holgadamente a `Arm 3` y `Arm 4`, se verifica empíricamente la hipótesis del subespacio universal determinista: **la geometría interna del modelo base define el sistema de proyección óptimo independientemente del corpus de ajuste**.

---

### EXP-19: Fusión Espectral Directa de Modelos (Spectral Core Model Merging / Task Arithmetic)

#### 1. Pregunta Científica
¿Permite el espacio de representación espectral compartido sumar adaptadores de tareas distintas mediante aritmética lineal de núcleos ($S_{\text{merge}} = \lambda S_A + (1-\lambda) S_B$) eliminando la interferencia destructiva que sufren los adaptadores espaciales LoRA ($B_A A_A + B_B A_B$)?

#### 2. Hipótesis Falsable
**H19:** Debido a que todos los adaptadores Spec-RAMA operan sobre la misma base ortogonal canónica ($P_L, P_R, \text{DCT/Wavelet}$), la adición lineal directa de sus núcleos espectrales preserva las capacidades de ambas tareas con un factor de interferencia significativamente menor que la suma de adaptadores LoRA de bajo rango o métodos TIES/DARE.

#### 3. Diseño Experimental
* **Tareas de Especialización:**
  - Tarea A: Especialización en texto literario / enciclopédico (WikiText-2).
  - Tarea B: Especialización en tareas de estilo / sintaxis estructurada (Penn Treebank o traducción básica).
* **Entrenamiento de Adaptadores:**
  - Entrenar $\text{SpecRAMA}_A$ y $\text{SpecRAMA}_B$ ($32\times 32$, 384 KB cada uno) sobre la misma base NF4.
  - Entrenar $\text{LoRA}_A$ y $\text{LoRA}_B$ ($r=4$, 2.25 MB cada uno) con hiperparámetros óptimos auditados.
* **Operaciones de Fusión:**
  - **Fusión Espectral:**
    $$S_{\text{merge}} = 0.5 \cdot S_A + 0.5 \cdot S_B$$
    $$\Delta W_{\text{merge}} = \mathcal{T}^{-1}(S_{\text{merge}})$$
  - **Fusión LoRA Directa:**
    $$\Delta W_{\text{LoRA}} = 0.5 \cdot (B_A A_A + B_B A_B)$$
  - **Fusión LoRA SVD / Ties-Merging:** Re-aproximación por SVD de rango $r=4$.
* **Métricas:** Perplejidad cruzada en Dataset A y Dataset B, degradación frente a los adaptadores individuales ($\Delta \text{bpt}_{\text{merge}} - \text{bpt}_{\text{single}}$).

---

### EXP-20: Subespacio Espectral Compartido Multi-Capa (`SharedSpecRAMA` Sub-10 KB)

#### 1. Pregunta Científica
¿Puede un único núcleo espectral maestro ser compartido por todas las capas de atención y MLP de la red, modulado únicamente por escalares o vectores diagonales por capa, alcanzando una adaptación de alta fidelidad con menos de 10 KB de parámetros totales?

#### 2. Hipótesis Falsable
**H20:** En concordancia con el Teorema del Regulador Confortable y la auto-similitud entre capas de transformadores, los desplazamientos de pesos entre capas comparten el mismo núcleo espectral de correlación. Un adaptador `SharedSpecRAMA` de $<10\text{ KB}$ retiene más del 90% de la capacidad de recuperación de precisión de Spec-RAMA completo (384 KB).

#### 3. Diseño Experimental
* **Arquitectura:** Usar `SharedSpecRAMALinear` (completamente refactorizado tras la corrección de la fuga de parámetros locales en la auditoría técnica).
* **Parámetros:**
  - Un único par de núcleos maestros compartidos a nivel de modelo:
    $$S_m^{\text{master}} \in \mathbb{R}^{32 \times 32}, \quad S_a^{\text{master}} \in \mathbb{R}^{32 \times 32} \quad (8\text{ KB total})$$
  - Por cada una de las 48 capas: vectores diagonales de modulación:
    $$\mathbf{v}_l^m \in \mathbb{R}^{32}, \quad \mathbf{v}_l^a \in \mathbb{R}^{32} \quad (48 \times 2 \times 32 \times 4\text{ bytes} \approx 12.2\text{ KB})$$
  - Variante Escalar: Un escalar por capa $\gamma_l \in \mathbb{R}$ ($48 \times 2 \times 4\text{ bytes} \approx 384\text{ bytes}$).
* **Presupuesto Total:**
  - Variante Vectorial: $\approx 20\text{ KB}$.
  - Variante Escalar: **$< 9\text{ KB}$** (récord absoluto de compresión).
* **Comparativas:**
  - Base FP32 sin adaptar (46.18 PPL).
  - Base NF4 sin adaptar (49.34 PPL).
  - VeRA (Kopiczko et al., 2024) en presupuesto iso-paramétrico.
  - Spec-RAMA completo (384 KB).

---

### EXP-21: Adaptación Espectral en Visión (ViT en CIFAR-10/100 con DirectML)

#### 1. Pregunta Científica
¿Se traslada la eficiencia de compresión espectral guiada por TSP a las capas de auto-atención en Vision Transformers (ViT), donde las correlaciones espaciales 2D de las imágenes inducen redundancias de canal aún más marcadas que en texto?

#### 2. Hipótesis Falsable
**H21:** En Vision Transformers (`vit-base-patch16-224` o `vit-tiny-patch16-224`), la ordenación por TSP de los canales de proyección de parches concentra más del 95% de la energía en el cuadrante de baja frecuencia, permitiendo que Spec-RAMA iguale la precisión de clasificación top-1 de LoRA ($r=4$) en CIFAR-10/100 con una reducción de tamaño de checkpoint de al menos $5\times$.

#### 3. Diseño Experimental
* **Modelo Base:** Vision Transformer (`vit-base-patch16-224` o `vit-tiny-patch16-224`) preentrenado en ImageNet.
* **Datasets:** CIFAR-10 y CIFAR-100 (adaptación out-of-domain).
* **Entorno de Ejecución Local:** AMD Ryzen 7 8845HS con 64 GB RAM (ejecución mediante CPU multihilo o `torch-directml` en Radeon 780M).
* **Brazos:**
  1. Full Fine-Tuning (todos los parámetros entrenables).
  2. Standard LoRA ($r=4$ en proyecciones Q, K, V y MLP).
  3. Spec-RAMA Wavelet ($32 \times 32$ con escala Parseval).
  4. Spec-RAMA DCT-2D ($32 \times 32$).
* **Métricas:** Top-1 Accuracy (%), Parámetros entrenables, Tamaño del adaptador en disco (KB), Consumo máximo de RAM/VRAM durante entrenamiento.

---

### EXP-22: Protocolo de Evaluación a Escala Completa y Benchmarks Downstream

#### 1. Pregunta Científica
¿Se mantienen idénticos los deltas relativos de perplejidad y bits por token medidos en EXP-11 cuando se evalúa sobre la totalidad de los 280.000 tokens de WikiText-2 y en tareas de clasificación estructurada (GLUE)?

#### 2. Hipótesis Falsable
**H22:** La ventaja relativa de Spec-RAMA medida en los primeros 100 bloques (25.600 tokens) se extrapola sin variaciones significativas ($\Delta \text{bpt} \le \pm 0.02$) al conjunto de test completo (280.000 tokens), y las ganancias de entropía se traducen directamente en precisión de clasificación equivalente a LoRA en GLUE (SST-2, MRPC, RTE).

#### 3. Diseño Experimental
* **Evaluación Completa WikiText-2:**
  - Evaluar los checkpoints guardados de EXP-11 y EXP-12 sobre el 100% de los tokens de `Salesforce/wikitext-2-raw-v1` test (sin truncar a `max_test_samples=100`).
* **Downstream Tasks (GLUE Benchmark):**
  - Fine-tuning sobre SST-2 (clasificación de sentimiento binaria).
  - Fine-tuning sobre MRPC (equivalencia semántica de oraciones).
  - Fine-tuning sobre RTE (inferencia de lenguaje natural).
* **Instrumentación:**
  - Serialización directa a JSON (`exp22_full_eval_results.json`).
  - Registro explícito de tokens evaluados, accuracy, F1-score y PPL.

---

## 3. Matriz de Priorización y Recursos

| Experimento | Pregunta Clave | Prioridad | Coste Computacional | Viabilidad Local (Ryzen 7 8845HS) | Dependencias |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **EXP-18** (Universalidad TSP) | ¿Es el sistema TSP invariante a la tarea? | **ALTA (P1)** | Bajo-Medio (~30-60 min CPU) | **100% Factible** (CPU/DirectML) | Código de permutación existente |
| **EXP-19** (Fusión Espectral) | ¿Es el merge espectral sin interferencia? | **ALTA (P1)** | Bajo (inferencia y suma directa) | **100% Factible** (Solo eval PPL) | Checkpoints de EXP-18 |
| **EXP-20** (`SharedSpecRAMA`) | ¿Puede adaptarse todo el modelo con <10 KB? | **MEDIA (P2)** | Bajo (~30 min CPU) | **100% Factible** | Refactor de `shared_layers.py` |
| **EXP-21** (ViT en Visión) | ¿Funciona Spec-RAMA en parches 2D de imagen? | **MEDIA (P2)** | Medio (~1-2 horas en DirectML) | **100% Factible** (PyTorch / DirectML) | Integración `timm` / ViT |
| **EXP-22** (WikiText Completo + GLUE)| ¿Se mantiene la ventaja en 280K tokens y GLUE? | **ALTA (P1)** | Bajo-Medio (solo re-evaluación) | **100% Factible** | Checkpoints existentes de EXP-11/12 |

---

## 4. Estándares Metrológicos de Ejecución (Obligatorios)

Para evitar la reaparición de artefactos experimentales como el observado en versiones preliminares de LoRA:

1. **Blindaje de Congelación:** Todo script debe invocar de forma obligatoria una función de validación de congelación antes de pasar el modelo al optimizador:
   ```python
   def assert_strictly_frozen_base(model, allowed_substrings):
       for name, param in model.named_parameters():
           if any(sub in name for sub in allowed_substrings):
               assert param.requires_grad, f"Parametro esperado entrenable no lo es: {name}"
           else:
               assert not param.requires_grad, f"Fuga de gradiente en parametro base: {name}"
   ```
2. **Conteo Dinámico Real:** Prohibido escribir números de parámetros "hardcodeados" en logs o tablas. El número de parámetros entrenables y el tamaño en bytes deben calcularse en tiempo de ejecución:
   ```python
   trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   trainable_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
   ```
3. **Serialización Estructurada:** Todo benchmark debe generar un archivo `.json` idéntico al estándar de `exp11_results.json` y `exp12_results.json`, con metadatos completos de `eval_tokens`, `eval_blocks`, `learning_rate` y `bits_per_token`.
4. **Semillas Múltiples:** Todo resultado final reportado debe incluir media y desviación estándar sobre al menos 3 semillas aleatorias (`seed=42, 1337, 2026`).

---

## 5. Referencias

1. Kaushik et al., "The Universal Weight Subspace Hypothesis: Overparameterization Creates Low-Dimensional Shared Representations" (arXiv:2512.05117, Diciembre 2025).
2. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022).
3. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (NeurIPS 2023).
4. Gao et al., "FourierFT: Parameter-Efficient Fine-Tuning via Discrete Fourier Transform" (ICML 2024).
5. Kopiczko et al., "VeRA: Vector-based Random Matrix Adaptation" (ICLR 2024).
6. Carbonell Martínez, M.R., "The Compact Representation Principle in Neural Architectures" (Agosto 2026).
