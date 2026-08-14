# Resumen Ejecutivo de la Auditoría Externa de Spec-RAMA

Este documento consolida y sintetiza de manera exhaustiva todos los hallazgos técnicos, metodológicos y teóricos identificados en las 4 rondas de la auditoría externa (`auditoria_externa.md`). Sirve como base técnica y hoja de ruta para la refactorización y blindaje experimental del framework `spec-rama`.

---

## 1. Diagnóstico Global y Veredicto

* **Valoración Conceptual**: **Muy Alta (8.5 / 10 en Originalidad, 5 / 5 en Honestidad Intelectual)**. La idea central —reordenar canales mediante permutaciones TSP bipartitas para concentrar energía y adaptar un núcleo espectral ultra-compacto— es genuinamente innovadora, falsable y con alto potencial para PEFT en presupuestos extremos ($<30$ KB).
* **Diagnóstico Crítico**: El arnés experimental contenía **errores de instrumentación, falta de congelación de capas base en los controles y discrepancias de contabilidad**, lo que distorsionaba varias conclusiones publicadas (tanto a favor como en contra de SpecRAMA).
* **Dirección de los Errores**: Múltiples fallos perjudicaban a SpecRAMA (ej. competir contra un LoRA artificialmente inflado con fine-tuning de embeddings, o subestimar la ventaja de parámetros como $5.2\times$ cuando en realidad es $6.0\times$).

---

## 2. Hallazgos Críticos y Bloqueantes (Prioridad Alta)

### 🔴 2.1. Bug Bloqueante en `permutation.py` (`TypeError`)
* **Archivo**: `spec_rama/permutation.py` (Línea 129).
* **Problema**: El despachador `compute_2d_permutations` invocaba:
  ```python
  return compute_joint_bipartite_2d_permutation(weight, num_iters=2)
  ```
  mientras que la firma de `compute_joint_bipartite_2d_permutation` define `max_iters: int = 5, tol: float = 1e-3`.
* **Impacto**: Cualquier llamada a `SpecRAMALinear(..., permutation_method="bipartite_tsp")` en HEAD crasheaba de inmediato con `TypeError: unexpected keyword argument 'num_iters'`. Demuestra que `exp11` no se ejecutó en la versión exacta de HEAD sin corregir ese argumento a `max_iters=2`.

---

### 🔴 2.2. Fuga de Parámetros no Congelados en LoRA (`exp11` y `exp12`): 40M Parámetros Entrenados
* **Archivos**: `benchmarks/exp11_proper_qlora_nf4_baseline.py` (Brazo 5) y `benchmarks/exp12_lora_hyperparameter_sweep.py`.
* **Problema**:
  * En los brazos SpecRAMA (2, 4, 7), se congelaba explícitamente todo el modelo antes de abrir solo los núcleos:
    ```python
    for p in model.parameters(): p.requires_grad = False
    for m in inject_spec: ...
    ```
  * En el brazo LoRA (Brazo 5 de `exp11` y todo `exp12`), se llamaba a `inject_lora_in_model(...)` **sin congelar el modelo base previamente**.
  * `LoRALinear.__init__` solo congela la capa lineal que envuelve. Por tanto, los siguientes tensores mantuvieron `requires_grad=True`:
    * `transformer.wte` (Word Token Embeddings, atado a `lm_head`): **38.597.376 parámetros**.
    * `transformer.wpe` (Position Embeddings): **786.432 parámetros**.
    * LayerNorms (`ln_1`, `ln_2`, `ln_f`): **38.400 parámetros**.
    * **Total de parámetros activos en LoRA: 40.012.056 (~40M)** frente a los 589K previstos.
* **Impacto Teórico y Experimental**:
  1. El resultado de LoRA $r=4$ a $lr=1e-4$ (**35.58 PPL**) no era LoRA puro, sino **LoRA + fine-tuning completo de la tabla de embeddings de salida**.
  2. El colapso de LoRA a $lr=1e-2$ (**391.52 PPL**) no se debía a "inestabilidad de matrices de bajo rango espaciales vs Parseval", sino a optimizar 38.6M de embeddings a un LR excesivo de $0.01$ con AdamW.
  3. **Conclusión**: El barrido de EXP-12 evaluaba la sensibilidad al LR del fine-tuning de embeddings, no de LoRA. SpecRAMA competía en desventaja contra un baseline dopado.

---

### 🔴 2.3. Tamaño Real del Test Set: 25.600 tokens vs 280.000 tokens
* **Archivos**: `benchmarks/exp11_proper_qlora_nf4_baseline.py` (y scripts previos).
* **Problema**: `prepare_wikitext_data` selecciona únicamente los primeros 100 bloques:
  ```python
  max_test_samples = 100
  block_size = 256
  test_data = lm_datasets["test"].select(range(min(len(lm_datasets["test"]), max_test_samples)))
  ```
  $100 \text{ bloques} \times 256 \text{ tokens} = \mathbf{25.600\text{ tokens}}$ (~9% del test set total de WikiText-2).
* **Impacto**: Las publicaciones y el README afirmaban "WikiText-2 Test Set (280.000 tokens)". Aunque la comparativa interna entre brazos es válida al usar los mismos 100 bloques, la métrica no cubre el test set completo.

---

## 3. Contabilidad de Parámetros y Tamaño del Adaptador

### 🟠 3.1. Cobertura Real de Módulos (48 vs 36) y Ratios Corregidos
* **Problema**: Al usar `targets = ["c_attn", "c_proj", "c_fc"]`, `"c_proj"` coincide con `attn.c_proj` y con `mlp.c_proj`. En GPT-2 Small (12 capas) se inyectan **48 módulos**, no 36.
* **Cálculo Real**:
  * **SpecRAMA Wavelet ($32\times 32$, 2 núcleos por módulo $M$ y $A$)**:
    $$48 \times 2 \times 32 \times 32 \times 4\text{ bytes} = 393.216\text{ bytes} = \mathbf{393.2\text{ kB}}$$
    *(El README reportaba 294.9 KB, que corresponde a 36 módulos)*.
  * **LoRA $r=4$ (48 módulos)**:
    $$48 \times (4 \times d_{\text{in}} + d_{\text{out}} \times 4) \times 4\text{ bytes} = 589.824\text{ params} = \mathbf{2.359.296\text{ bytes}} = \mathbf{2.36\text{ MB}}$$
    *(El README reportaba 1.55 MB, correspondiente a 36 módulos)*.
  * **Ratio Real de Parámetros**:
    $$\frac{2.36\text{ MB}}{393.2\text{ kB}} = \mathbf{6.0\times\text{ reducción}}$$
    *(Superior al $5.2\times$ anunciado)*.

---

### 🟠 3.2. Tamaño del Checkpoint (`state_dict`) por Buffers de Permutación
* **Problema**: `SpecRAMALinear` registra `row_perm`, `col_perm`, `row_inv_perm` y `col_inv_perm` como `register_buffer` en tipo `torch.long` (int64).
  * Por bloque: $12.288\text{ índices} \times 2 \times 8\text{ bytes} = 196.608\text{ B}$.
  * En 12 bloques: **2.36 MB en buffers de índices**.
  * El `state_dict` completo guardado en disco pesa **2.75 MB** (mayor que el de LoRA).
* **Solución**:
  1. Las permutaciones son 100% deterministas y reproducibles a partir del peso base $W_0$. No necesitan persistirse en el archivo de pesos del adaptador.
  2. Si se persisten, deben almacenarse en `int16` (hasta dimensión 32.767) y solo las directas (las inversas se calculan con `argsort`).

---

## 4. Rigor en Cuantización y Metodología

### 🟠 4.1. Cuantización Simulada ("Fake Quantization") vs Runtime NF4 Real
* **Problema**: `quantize_blockwise_nf` aplica la cuantización de cuantiles NF4 por bloques (`block_size=64`) pero des-cuantiza inmediatamente a `float32`.
* **Impacto**:
  * Es una simulación matemática matemáticamente correcta de la perturbación/ruido de cuantización NF4.
  * No utiliza tensores empaquetados de 4 bits, kernels de `bitsandbytes` ni reduce el consumo real de VRAM en memoria (el modelo sigue en FP32, ocupando ~200 MB con embeddings).
  * **Acción**: Clarificar en la documentación que se trata de emulación de daño de cuantización NF4, reservando el término "4-bit runtime" para cuando se integre `Linear4bit` real.

---

### 🟠 4.2. Discrepancia en Scripts y Salidas Hardcodeadas
* **Problema**:
  * En `exp11`, la tabla final de print contiene literales fijos (`-0.293 bpt`, `294.9 KB`, `174.6 KB`) en lugar de variables computadas.
  * No había serialización a JSON o CSV; los datos se transcribían a mano a los Markdown.
  * `count_trainable_parameters` y `merge_spec_rama_modules` se importaban pero no se utilizaban.

---

## 5. Fundamento Teórico: Espectral vs Esparsidad Estructurada

### 💡 5.1. Naturaleza de Haar Wavelet 2D (1 Nivel)
* **Hallazgo**: `haar_dwt_2d` / `haar_idwt_2d` es una transformada de Haar de un solo nivel (soporte compacto $2\times 2$).
* Inyectar un núcleo $32\times 32$ en la esquina de la banda $LL$ sintetiza una actualización no nula únicamente en el bloque superior izquierdo de $64\times 64$ de la matriz permutada.
* En `mlp.c_fc` ($3072 \times 768$), $64\times 64$ afecta solo al **0.17% de las entradas**.
* **Reinterpretación**: Esto no debilita el resultado, lo hace más interesante: **SpecRAMA Wavelet actúa como una técnica de esparsidad estructurada ultra-localizada guiada por TSP**.
* En contraste, la rama **DCT-2D** sí es una parametrización de bajo rango ortogonal global denso.

---

### 💡 5.2. Presunta Invariancia de Parseval y Barrido de Sensibilidad
* **Hallazgo**: El factor de escala $\frac{\alpha}{\sqrt{k_{\text{out}} \cdot k_{\text{in}}} \cdot \text{std}(W_0)}$ estabiliza numéricamente el gradiente.
* Sin embargo, para sostener la tesis de "invarianza frente a hiperparámetros", se debe ejecutar el barrido completo de $lr \in [10^{-4}, 10^{-2}]$ tanto para LoRA (congelado) como para SpecRAMA bajo el mismo protocolo.

---

## 6. Bugs Menores y Deuda Técnica en Módulos

1. **`spec_rama/shared_layers.py`**:
   * `SharedSpecRAMALinear` instanciaba internamente `SpecRAMALinear`, creando `core_m` y `core_a` locales que quedaban como parámetros no usados con `requires_grad=True`. Esto inflaba el conteo a 18.8 KB en lugar de los ~800 bytes del diseño Sub-KB.
2. **`spec_rama/transforms.py` (`walsh`)**:
   * `get_walsh_matrix_1d` truncaba matrices de Hadamard de tamaño $2^{\lceil \log_2 N \rceil}$ a $N$, rompiendo la propiedad de ortogonalidad ($H H^\top \neq I$) para dimensiones como 768, 2304 o 3072.
3. **`benchmarks/exp12_lora_hyperparameter_sweep.py`**:
   * Definía `alphas = [4.0, 8.0, 16.0]` pero el bucle solo ejecutaba `for alpha in [8.0]`.
4. **`spec_rama/layers.py` (`w_std`)**:
   * `self.w_std` no era registrado como buffer (`register_buffer`), lo que provocaba que se recalculase al recargar checkpoints.

---

## 7. Plan de Acción Inmediato (Checklist de Refactorización)

- [x] **1. Corregir `permutation.py`**: Cambiar `num_iters=2` por `max_iters=2` en `compute_2d_permutations`.
- [x] **2. Blindar la Congelación en LoRA**: Asegurar que `inject_lora_in_model` o el script de benchmark congele explícitamente `model.parameters()` antes de habilitar gradientes solo en `lora_A` y `lora_B`.
- [x] **3. Instrumentación Limpia y Transparente**:
  - Imprimir conteo medido de parámetros con `requires_grad=True` en todos los brazos.
  - Imprimir tamaño exacto en bytes de los tensores entrenables.
  - Serializar resultados a un archivo `results_expXX.json`.
- [x] **4. Arreglar `shared_layers.py`**: Eliminar los parámetros locales no utilizados en `SharedSpecRAMALinear` para alcanzar la promesa real Sub-KB.
- [ ] **5. Corregir Documentación y Metadatos**:
  - Actualizar tamaños reales: SpecRAMA $32\times 32$ (48 mód) = 393.2 kB; LoRA $r=4$ (48 mód) = 2.36 MB ($6.0\times$ ratio).
  - Clarificar tokens evaluados (especificar si es subset de 25.6k o test completo de 280k).
  - Documentar la naturaleza de fake-quantization en NF4.
- [ ] **6. Re-ejecutar EXP-11 y EXP-12**: Obtener los números definitivos, limpios y reproducibles.
