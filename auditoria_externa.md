# Evaluación — Ronda 3 (final)
## `transforms.py` + `permutation.py` + `exp11_proper_qlora_nf4_baseline.py`

---

## Veredicto final

**Contratar condicionalmente, con take-home obligatorio y bloqueante.** No contratar sin él.

Esta ronda cambia las cosas de forma importante. Confirmo §A.1 de la ronda 2 (el método no es espectral en su configuración estrella), pero además encuentro **cuatro problemas nuevos, cada uno suficiente por sí solo para invalidar el resultado titular**, incluido uno que impide directamente que el experimento se ejecute desde el HEAD del repositorio.

| Eje | R1 | R2 | R3 | |
|---|:---:|:---:|:---:|---|
| Originalidad | 4,5 | 4,5 | **4,5** | = |
| Ejecución de ingeniería | 4 | 4 | **2,5** | ↓↓ El arnés estrella no ejecuta |
| Rigor experimental | 2,5 | 3 | **2** | ↓ Baselines con cobertura desigual |
| Honestidad intelectual | 4,5 | 5 | **5** | = Sigue siendo su mejor activo |
| Comunicación / disciplina | 2 | 1,5 | **1** | ↓ Cifras clave del abstract, mal |
| Comprensión de su método | — | 2,5 | **2,5** | = |

Y una nota importante: **me equivoqué en la ronda 2** sobre los coeficientes alpha. Lo corrijo en §7.

---

## 1. 🔴 BLOQUEANTE: el experimento estrella no puede ejecutarse desde HEAD

`permutation.py`, línea del despachador:

```python
if method == "bipartite_tsp":
    return compute_joint_bipartite_2d_permutation(weight, num_iters=2)
```

Y la firma de la función:

```python
def compute_joint_bipartite_2d_permutation(
    weight: torch.Tensor,
    max_iters: int = 5,
    tol: float = 1e-3
) -> Tuple[torch.Tensor, torch.Tensor]:
```

**No existe el parámetro `num_iters`.** Python lanza `TypeError: compute_joint_bipartite_2d_permutation() got an unexpected keyword argument 'num_iters'`.

Y `exp11` pasa `permutation_method="bipartite_tsp"` en los **cuatro** brazos de SpecRAMA (2, 4, 7 ×2). Es decir:

> **El código publicado del benchmark insignia lanza una excepción en la primera línea del primer módulo que envuelve.**

Las implicaciones son severas y no admiten interpretación benévola:

1. **Los resultados de EXP-11 no fueron producidos por este código.** Fueron producidos por una versión anterior que ya no está en el repositorio, o el commit que introdujo el bug se subió después sin volver a correr nada.
2. El README dice `modal run modal_runner.py --exp exp11` bajo el epígrafe **"Reproducible Benchmarks"**. No lo es. Nadie ha ejecutado ese comando desde que existe esta versión del fichero.
3. **No hay CI, ni un smoke test, ni una sola ejecución de verificación post-commit.** Un `pytest` de tres líneas que instanciara un `SpecRAMALinear` con cada `permutation_method` lo habría detectado.
4. Y lo más incómodo: la permutación bipartita alternante es **la novedad nominal del método** — la que aparece en el título, en el diagrama y en las dos figuras del README. Es el único camino de código que está roto.

Esto reordena la evaluación entera. En las rondas 1 y 2 el problema era "escribe el abstract antes de las ablaciones". Aquí el problema es **"publica como reproducible algo que no ha ejecutado"**. Es una categoría distinta.

---

## 2. 🔴 §A.1 confirmado: no es espectral, es esparsidad estructurada

`transforms.py` cierra la duda que dejé abierta:

```python
def haar_idwt_2d(LL, LH, HL, HH, target_shape=None):
    H_out = LL.shape[0] * 2
    ...
    out[0::2, 0::2] = x00   # ← soporte compacto 2×2, un solo nivel
```

**Un nivel. Sin recursión.** Confirmo la ronda 2 y retiro la reserva.

Con el core en `LL[:k_out, :k_in]` y LH=HL=HH=0, la matriz sintetizada $M_{\text{perm}}$ es no nula **solo en su bloque $2k_{\text{out}} \times 2k_{\text{in}}$ superior izquierdo**. Los números exactos para EXP-11 (core 32×32, 48 módulos):

| Módulo | Forma | Entradas no nulas | Fracción |
|---|---:|---:|---:|
| `c_attn` | 2304×768 | 64×64 | **0,23%** |
| `attn.c_proj` | 768×768 | 64×64 | **0,69%** |
| `mlp.c_fc` | 3072×768 | 64×64 | **0,17%** |
| `mlp.c_proj` | 768×3072 | 64×64 | **0,17%** |

Y para el brazo asimétrico con core 8×8 en las capas MLP: **16×16 sobre 3072×768 = 0,011%**.

Con eso, la cadena de afirmaciones se cae entera:

- **"2D *Multiscale* DWT"** → un solo nivel. Falso.
- **"Parseval Energy Scaling"** → no hay energía global que conservar; hay un parche local.
- **"spectral energy equalization over a compact 2D core"** → no ecualiza nada. Selecciona 64 canales de salida × 64 de entrada (los primeros del orden TSP) y les aplica un update denso a resolución de bloque 2×2.
- **La permutación TSP no es "concentración de energía"**, es un **criterio de selección de canales**. Que es lo mismo que decir que la ablación con permutación aleatoria = selección aleatoria de 64 canales.
- El estado del arte de referencia (FourierFT, VeRA) es el equivocado. El correcto es SpIEL, LoRA-dispersa, adaptadores por canal.

**Y, otra vez: esto hace el resultado más interesante, no menos.** Si tocando el 0,23% de las entradas se pasa de 49,34 a 37,98 PPL, eso es un hallazgo sobre esparsidad estructurada que merece publicarse. Con otro título, otro marco teórico y otras comparaciones.

**Confirmación adicional de la rama DCT.** `get_dct_matrix_1d` devuelve $D$ ortonormal, y `_synthesize_matrix_from_core` hace $D_{\text{out}}[:k]^\top S\, D_{\text{in}}[:k]$. Eso es **exactamente un update de rango $\le k$ con bases congeladas** — es decir, LoRA con $A$ y $B$ fijas y solo el core $k\times k$ entrenable. Global, denso, y conceptualmente muy distinto de la rama wavelet. El experimento DCT-vs-Wavelet a igual $k$ sigue siendo la prueba decisiva, y ahora sé exactamente qué contrasta: **rango bajo global vs esparsidad estructurada local.**

---

## 3. 🔴 El baseline de LoRA: dos problemas, uno potencialmente fatal

### 3.a Posible falta de congelación (necesito `lora_baseline.py`)

Compárense los brazos. SpecRAMA (2, 4, 7):

```python
model = model.to(device)
for p in model.parameters(): p.requires_grad = False      # ← congelación explícita
for m in inject_spec:
    if m.core_m is not None: m.core_m.requires_grad = True
    if m.core_a is not None: m.core_a.requires_grad = True
```

LoRA (brazo 5):

```python
inject_lora_in_model(model_nf4_lora, target_modules=[...], rank=4, alpha=8.0)
model_nf4_lora = model_nf4_lora.to(device)
train_on_dataset_long_horizon(model_nf4_lora, ...)          # ← sin congelación explícita
```

Y el optimizador es `AdamW(filter(lambda p: p.requires_grad, model.parameters()))`. **Si `inject_lora_in_model` no congela el modelo base, el brazo 5 no es LoRA: es fine-tuning completo de 124M parámetros** partiendo de pesos NF4-dequantizados, a `lr=1e-4`, 500 pasos.

Eso explicaría *ambos* resultados de LoRA de forma perfectamente coherente:
- a `lr=1e-4` → 35,58 PPL (fine-tuning completo funciona muy bien)
- a `lr=1e-2` → 391,52 PPL (fine-tuning completo diverge espectacularmente)

No puedo confirmarlo sin el fichero. Pero la asimetría estructural entre brazos — congelación explícita en uno, implícita en el otro — es exactamente el patrón donde vive este bug. **Es lo primero que hay que verificar.**

Nótese la ironía: en la ronda 1 el problema era un baseline de LoRA **artificialmente débil**. Aquí puede ser uno **artificialmente fuerte**. Errores en direcciones opuestas. Eso es tranquilizador sobre la intención y demoledor sobre el proceso: no hay sesgo motivado, hay falta de disciplina de control.

### 3.b Los dos brazos no cubren las mismas capas

El comentario del código dice `Standard LoRA (r=4, 405K params)`. Reconstruyo:

| Configuración | Parámetros |
|---|---:|
| LoRA r=4 sobre `c_attn` + `attn.c_proj` + `mlp.c_fc` (36 módulos) | **405.504** ✓ coincide |
| LoRA r=4 sobre los 4 módulos por bloque (48 módulos) | 589.824 ✗ |

Pero SpecRAMA usa `any(t in name for t in ["c_attn","c_proj","c_fc"])`, y `"c_proj"` **coincide tanto con `attn.c_proj` como con `mlp.c_proj`**. Son **48 módulos**.

> **SpecRAMA adapta 48 matrices. LoRA adapta 36.** No es un head-to-head.

Y en particular LoRA no toca `mlp.c_proj`, que es una de las matrices grandes (3072×768). Es una ventaja no declarada a favor de SpecRAMA que corre en dirección contraria al titular de eficiencia.

---

## 4. 🔴 "280.000 tokens" son 25.600

```python
def prepare_wikitext_data(tokenizer, block_size=256, max_train_samples=600, max_test_samples=100):
...
test_data = lm_datasets["test"].select(range(min(len(...), max_test_samples)))
```

$100 \text{ bloques} \times 256 \text{ tokens} = \mathbf{25.600\ tokens}$.

La cifra **280.000** aparece en el abstract del paper, en la cabecera del README, en la tabla de resultados, en la sección "Threats to Validity" y en el título del documento de findings. Es **11× la realidad**, y corresponde aproximadamente al test set *completo* de WikiText-2 — que es justo lo que el `.select(range(100))` impide usar. Se está evaluando sobre **~9% del test set**.

Esto no es cosmético. Con 25.600 tokens, una sola corrida y sin semillas, la diferencia 37,98 vs 35,58 PPL no tiene ningún respaldo estadístico. Y el claim del brazo 7 ("supera a FP32 nativo": 45,99 vs 46,18, un 0,4%) es indefendible sobre esa muestra.

De paso: el entrenamiento son 600 bloques = 153.600 tokens, y 500 pasos × 4 × 256 = 512.000 tokens vistos ≈ **3,3 épocas sobre 150k tokens**. Es un régimen minúsculo. Legítimo para prototipar; hay que decirlo.

---

## 5. 🟠 La contabilidad del adaptador: el 5,2× no existe

Tres errores encadenados.

**(a) El número base asume 36 módulos, el código inyecta 48.**
$36 \times 2 \times 1024 \times 4\text{ B} = 294.912\text{ B} \to$ "294,9 KB". Con los 48 reales: $48 \times 2 \times 1024 \times 4 = \mathbf{393.216\ B}$.

**(b) Unidades mezcladas entre los dos términos de la comparación.** "294,9 KB" usa kB decimal ($/1000$); "1,55 MB" para LoRA usa MiB binario ($405.504 \times 4 / 2^{20} = 1{,}547$). En el mismo README aparece además "288 KB" para el mismo adaptador — que es la misma cifra en KiB. Tres unidades distintas para dos números que se dividen entre sí.

**(c) Las permutaciones son `register_buffer`, por tanto están en el `state_dict`.** Van en el checkpoint, sí o sí. En `torch.long`: por bloque, $(2304{+}768) + (768{+}768) + (3072{+}768) + (768{+}3072) = 12.288$ índices; ×2 (perm + inversa) ×8 B = 196.608 B; ×12 bloques = **2,36 MB**.

Tabla corregida, todo en bytes decimales:

| Contabilidad | SpecRAMA | LoRA r=4 | Ratio |
|---|---:|---:|---:|
| Como se reporta | 294,9 kB | 1,55 MB | 5,2× |
| Cores, cobertura real (48 mód.) | 393,2 kB | 1,62 MB | **4,1×** |
| **`state_dict` tal como se guarda hoy** | **2,75 MB** | 1,62 MB | **0,59× (peor)** |
| Cores + perms int16, sin inversas | 688 kB | 1,62 MB | **2,4×** |

La defensa válida sigue siendo que la permutación es recomputable desde $W_0$. Y aquí hay buena noticia: `compute_greedy_tsp_1d` se ejecuta sobre el peso **en CPU** en el constructor (el modelo se envuelve antes de `.to(device)`), con `torch.cdist` determinista y nodo inicial fijo. Es reproducible. **Pero hay que implementar la regeneración y decirlo**, y la fila honesta hoy es la tercera: **el adaptador guardado es 1,7× más grande que el de LoRA.**

---

## 6. 🟠 Los 65,8 MB / 86,1% de ahorro no corresponden a lo que se ejecutó

`quantize_blockwise_nf` es una implementación correcta de cuantización NF por bloques con absmax — pero **devuelve FP32 dequantizado**. Se confirma §A.2: cuantización simulada, cero tensores de 4 bits, cero `bitsandbytes`.

Peor: el comentario del código dice explícitamente

```python
# Target linear projections ONLY (keeping embeddings wte/wpe and lm_head in FP16 as per QLoRA protocol)
```

pero **nada se pasa a FP16**. Y los embeddings de GPT-2 son 39,4M de 124M parámetros (**32%**). Contabilidad real de lo ejecutado:

| Componente | Params | A 4 bits | En FP32 |
|---|---:|---:|---:|
| Lineales del transformer | 85 M | 42,5 MB | — |
| Embeddings + wpe + LN + bias | 39 M | — | 157,6 MB |
| **Total** | 124 M | | **≈ 200 MB** |

vs 496 MB en FP32 → **~60% de ahorro**, no 86,1%. Para llegar a 65,8 MB harían falta ~4,25 bits/parámetro de media sobre *todo* el modelo, embeddings incluidos. Es decir: **la cifra describe un modelo que nunca se evaluó.**

Y no hay una sola llamada a `torch.cuda.max_memory_allocated()` en el fichero. Es un número calculado a mano sobre una configuración hipotética, presentado como medición.

---

## 7. ⚠️ Corrección de mi propio error (ronda 2)

En la ronda 2 escribí que `alpha_a=8.0` vs `alpha_m=1.0` sugería que "la vía multiplicativa puede ser vestigial". **Eso era incorrecto y lo retiro.**

Eran los *defaults* de `layers.py`. `exp11` pasa explícitamente `alpha_m=8.0, alpha_a=2.0`, es decir, lo contrario. Y hay un segundo factor que no consideré: la vía multiplicativa divide además por `w_std`. Con $k=32$ y $\text{std}(W_0) \approx 0{,}14$ para GPT-2:

$$\text{escala}_m = \frac{8{,}0}{32 \times 0{,}14} \approx 1{,}79 \qquad \text{escala}_a = \frac{2{,}0}{32} = 0{,}0625$$

**La vía multiplicativa opera a ~29× la escala de la aditiva en el experimento real.** Mi inferencia estaba invertida. La ablación sigue siendo necesaria — ahora la hipótesis es que **la aditiva puede ser la vestigial**, y que el adaptador efectivo sea de 197 kB en lugar de 393 kB.

(Dejo esto aquí explícitamente porque es exactamente el comportamiento que le estoy pidiendo a él: cuando el dato nuevo refuta tu lectura anterior, la retiras por escrito.)

---

## 8. 🟡 Bugs latentes y desincronizaciones

**`get_walsh_matrix_1d` no es ortogonal para dimensiones que no sean potencia de 2.**
```python
p = 2 ** math.ceil(math.log2(N))
H_full = get_walsh_matrix_1d(p)
return H_full[:N, :N]     # ← una submatriz de Hadamard NO es ortogonal
```
Las dimensiones de GPT-2 son 768, 2304, 3072 — **ninguna es potencia de 2**. La rama `walsh` usaría una submatriz $768\times768$ de una Hadamard $1024\times1024$, que no cumple $H H^\top = I$. No afecta a EXP-11 (usa wavelet), pero invalida cualquier resultado futuro con `transform_type="walsh"` y rompe la exactitud del merge en esa rama. El arreglo correcto es proyectar sobre las primeras $N$ *filas* con reortogonalización, o hacer zero-padding a $p$ y truncar la salida.

**`haar_dwt_2d` crashea con dimensiones impares.** `F.pad(x, (0,pw,0,ph), mode='reflect')` sobre un tensor 2D lanza `NotImplementedError` en PyTorch (reflect 2D exige entrada 3D/4D). No se ejecuta en el forward, pero es código muerto roto.

**Número hardcodeado y obsoleto en la tabla de salida.** La línea de impresión del brazo 2 contiene literalmente `-0.293 bpt`, mientras el documento reporta `-0.360 bpt` (que es el valor correcto: $5{,}169 - 5{,}529$). Es decir: **la tabla que imprime el script y la tabla que está en el paper no coinciden**, y los tamaños de adaptador ("294.9 KB", "174.6 KB") también son cadenas literales, no calculadas. `count_trainable_parameters` y `merge_spec_rama_modules` se importan y **nunca se usan**.

**`w_std` sigue sin ser buffer** (§A.5 de la ronda 2): se recalcula al recargar checkpoint y reescala silenciosamente la vía multiplicativa — que ahora sabemos que es la dominante. El impacto de ese bug es mayor de lo que estimé.

**Confound menor pero real:** en el brazo 2 la permutación se calcula sobre pesos FP32; en el 4, sobre pesos NF4. Son permutaciones distintas. Defendible, pero no está documentado.

---

## 9. Diagnóstico final

En la ronda 2 dije que sus motores de generación y de verificación están **desacoplados en el tiempo**. Esta ronda precisa el diagnóstico y lo agrava:

> **La verificación se aplica a las *ideas* y casi nunca a los *artefactos*.**

Es capaz de la observación de V138 (Parseval anula su propia memoria holográfica) o de montar EXP-12 para auditar su propio titular de LoRA. Eso es verificación conceptual de primer nivel, y es genuina.

Pero **no verifica que el código corra, que los números del abstract salgan del script, que las unidades sean consistentes, que los brazos cubran las mismas capas, o que el tamaño del test set sea el que dice**. Los cinco fallos de esta ronda son todos de esa clase. Ninguno requiere talento para detectarse; requieren un `pytest`, un `assert`, y leer la tabla que imprime tu propio programa.

El fallo de la cobertura desigual (48 vs 36) y el posible fallo de congelación de LoRA son los más caros, porque el head-to-head es la afirmación central del trabajo y **no está controlado en ninguna de las dos direcciones**.

**Balance.** La honestidad sigue intacta y la mantengo en 5/5: nada de lo que he encontrado sugiere manipulación. Los errores van en ambas direcciones, incluidos varios que le perjudican (48 módulos frente a 36; el adaptador guardado más grande que LoRA). Es desorden, no fraude. Pero **el desorden a esta escala produce papers que no replican**, y en un lab de frontera eso cuesta reputación institucional, no solo tiempo.

---

## 10. Take-home (obligatorio, bloqueante) — 4 días

Le mando esto tal cual, sin pistas:

> *"Tu EXP-11 es el resultado central del repo. Antes de seguir: (1) haz que `modal run modal_runner.py --exp exp11` se ejecute de principio a fin desde un clon limpio; (2) instrumenta el script para que **imprima medidos**, no hardcodeados, el número de módulos adaptados, los parámetros entrenables por brazo, el tamaño en bytes del `state_dict` del adaptador, el número de tokens de evaluación y la VRAM pico; (3) añade tres semillas; (4) añade tres brazos de ablación: permutación identidad, permutación aleatoria y `transform_type='dct'` al mismo `k`. Mándame la tabla resultante y un párrafo diciendo qué de lo que afirmabas sigue en pie."*

**Criterios de evaluación:**

| Descubre por su cuenta | Peso |
|---|---|
| El `TypeError` de `num_iters` | Obligatorio. Si no lo encuentra, no puede ejecutar nada |
| Que el test set son 25.600 tokens, no 280.000 | Alto |
| Que SpecRAMA cubre 48 módulos y LoRA 36 | Alto |
| Si `inject_lora_in_model` congela o no la base | **Crítico** |
| Que el `state_dict` incluye 2,36 MB de permutaciones | Medio |
| Que la wavelet toca el 0,23% de las entradas | **El que más me interesa** |

**Señal de contratación inmediata:** que vuelva diciendo *"tenías razón en lo de la wavelet, es esparsidad estructurada; aquí está la tabla DCT-vs-wavelet-vs-aleatoria; el head-to-head con LoRA estaba mal controlado en dos sentidos; he reescrito el abstract y he retirado la versión anterior del repo."*

**Señal de rechazo:** que arregle el `TypeError`, vuelva a correr y mande los mismos números sin cuestionar nada más.

---

## 11. Preguntas de onsite (versión final, 6)

1. **«`compute_2d_permutations` llama a `compute_joint_bipartite_2d_permutation(weight, num_iters=2)`. ¿Qué pasa cuando ejecutas eso?»** — Diagnóstico puro. Y a continuación: «entonces, ¿con qué código se produjeron los números del README?»
2. **«En tu rama wavelet, LH=HL=HH=0 y el core va en la esquina de LL. La IDWT de Haar es de soporte compacto. ¿Qué fracción de $W$ es no nula en tu $M$?»** — La pregunta central. Si llega al 0,23% solo, todo lo demás es corregible.
3. **«Tu `prepare_wikitext_data` usa `max_test_samples=100` con `block_size=256`. ¿Cuántos tokens son? ¿Y qué dice tu abstract?»**
4. **«Los brazos 2, 4 y 7 congelan el modelo explícitamente. El 5 no. ¿Qué hace `inject_lora_in_model`?»**
5. **«`row_perm` y `col_perm` son buffers. ¿Están en el `state_dict`? ¿Cuánto pesan? ¿Sigue siendo 5,2×?»**
6. **«En V138 escribiste que por Parseval la FWHT no cambia el ranking por similitud, lo que anula tu propia memoria holográfica. ¿Cuándo te diste cuenta y por qué lo dejaste escrito?»** — Se mantiene desde la ronda 2. Es su mejor historia y quiero oírla.

---

## 12. Recomendación

**Onsite sí. Oferta solo tras el take-home.** Nivel: research engineer mid, en eficiencia de inferencia / cuantización / PEFT, con IC senior asignado y **revisión obligatoria de código antes de cualquier publicación externa durante los primeros 12 meses.**

**Lo que compramos.** Dos años de trayectoria coherente y trazable (v128 → v290 → spec-rama), 233 experimentos ejecutados de verdad, velocidad de prototipado excepcional, criterio para diseñar controles no obvios cuando decide diseñarlos, y —el activo que ninguna de las tres rondas ha erosionado— una disposición documentada y repetida a demoler sus propios resultados. Las notas de V138, el `[ANCLA-NEGATIVO]` de V87c, los seis `INVÁLIDO` de v292–v297 y la nota de auditoría de EXP-12 son evidencia real de carácter científico. Esa parte no se enseña.

**Lo que hay que arreglar, y ahora es más grande de lo que pensaba.** No es solo "escribe el abstract después de las ablaciones". Es: **ningún número entra en un documento si no lo imprime el script que lo produjo.** Test de humo por camino de código. Una unidad, un resultado, un fichero. Y baselines que cubran exactamente las mismas capas con exactamente el mismo presupuesto de búsqueda de hiperparámetros.

**Riesgo si contratamos.** Que la velocidad de generación nunca cristalice en un resultado defendible. Mitigable, pero requiere un senior dispuesto a bloquear publicaciones, no solo a comentarlas.

**Riesgo si no contratamos.** Sigue siendo real. Este perfil, con seis meses de disciplina impuesta, produce trabajo publicable. Los problemas que he encontrado son de proceso y son enseñables; la honestidad y la fecundidad de ideas no lo son. Preferiría contratarlo y tener que corregirle el proceso a tener el problema inverso.

**Mi voto: sí, condicionado al take-home.** Y le mandaría el take-home esta semana.

---

*Pendiente y ahora bloqueante: `spec_rama/lora_baseline.py` (§3.a). Es lo único que puede convertir "resultado mal contabilizado" en "resultado inválido". Después, `modal_runner.py` y `exp12`.*

*Higiene, tercera vez: fechas futuras («August 2026»), `findings_v12.md` referenciado y no listado, "10 inválidos" en la cabecera contra 6 en la tabla, y ahora un `(NEEDS CORRECTION!!!)` sin resolver en el abstract del paper. Sugiero que antes de la entrevista haga una pasada de auditoría documental completa — y que la trate como parte de la evaluación.*





----





# Evaluación — Ronda 4
## `lora_baseline.py` + `modal_runner.py` + `exp12_lora_hyperparameter_sweep.py`

---

## Veredicto

**Se mantiene: onsite sí, oferta solo tras take-home.** Pero la pregunta bloqueante de la ronda 3 (§3.a) ya tiene respuesta, y es la mala. Además debo **corregir un error propio** de la ronda 3, y esa corrección va en su contra por un lado y a su favor por otro.

Resultado neto: **el head-to-head central del proyecto no está roto en una dirección, está roto en las dos, y el valor correcto es desconocido.** Y —esto es lo nuevo y lo más serio— **EXP-12, la auditoría que en la ronda 1 califiqué como su mejor activo, hereda el mismo defecto y por tanto también es inválida.**

| Eje | R1 | R2 | R3 | R4 | |
|---|:---:|:---:|:---:|:---:|---|
| Originalidad | 4,5 | 4,5 | 4,5 | **4,5** | = |
| Ejecución de ingeniería | 4 | 4 | 2,5 | **2** | ↓ |
| Rigor experimental | 2,5 | 3 | 2 | **1,5** | ↓ El control principal no controla nada |
| Honestidad intelectual (voluntad) | 4,5 | 5 | 5 | **5** | = |
| Honestidad intelectual (eficacia) | — | — | — | **2** | Nuevo. Ver §7 |
| Comunicación / disciplina | 2 | 1,5 | 1 | **1** | = |
| Comprensión de su método | — | 2,5 | 2,5 | **2,5** | = |

---

## 1. 🔴 CONFIRMADO, y peor de lo que estimé: el brazo LoRA entrena 40M de parámetros

`LoRALinear.__init__` **sí** congela la capa que envuelve:

```python
self.base_layer.weight.requires_grad = False
if self.base_layer.bias is not None:
    self.base_layer.bias.requires_grad = False
```

Congela **solo los módulos que envuelve**. Todo lo demás conserva `requires_grad=True` por defecto, y el optimizador es `AdamW(filter(lambda p: p.requires_grad, ...))`.

Contraste directo entre brazos de `exp11`:

```python
# Brazos 2, 4, 7 (SpecRAMA)
for p in model.parameters(): p.requires_grad = False   # ← congela TODO, embeddings incluidos
for m in inject_spec: m.core_m.requires_grad = True    # ← reabre solo los cores

# Brazo 5 (LoRA)
inject_lora_in_model(...)                              # ← congela solo las 48 matrices envueltas
model_nf4_lora.to(device)
train_on_dataset_long_horizon(...)                     # ← el resto sigue entrenable
```

Qué queda entrenable en el brazo LoRA:

| Tensor | Parámetros | ¿Entrenable? |
|---|---:|:---:|
| Adaptadores LoRA | 589.824 | ✅ (previsto) |
| **`transformer.wte`** (atado a `lm_head`) | **38.597.376** | ❌ **no previsto** |
| **`transformer.wpe`** | **786.432** | ❌ **no previsto** |
| LayerNorms (`ln_1`, `ln_2`, `ln_f`) | 38.400 | ❌ **no previsto** |
| **Total entrenable** | **≈ 40.012.000** | |

> **El "baseline de LoRA r=4" es en realidad LoRA r=4 + fine-tuning completo de la matriz de embeddings y de la cabeza de lenguaje.** 40 M de parámetros entrenables, 68× la cifra reportada, y el 32% del modelo.

Y no es un tensor cualquiera: `wte` está atado a `lm_head`, así que se está reentrenando **la capa de salida completa** sobre WikiText-2. Es, con diferencia, la intervención de mayor impacto en perplejidad de dominio que existe en GPT-2. La comparación no es "adaptador de 294,9 KB vs adaptador de 1,55 MB"; es "adaptador de 393 KB vs adaptador + 154 MB de embeddings reentrenados".

**Explica los dos resultados de LoRA de forma exacta y sin residuo:**
- `lr=1e-4` → **35,58 PPL**. Ajustar embeddings a un corpus con LR conservador funciona extraordinariamente bien.
- `lr=1e-2` → **391,52 / 439,71 PPL**. Ajustar 38,6 M de embeddings a `1e-2` con AdamW diverge, como es de esperar.

La "divergencia por overshooting sobre matrices de bajo rango sin escalar" que él diagnostica en la nota de auditoría del README **no es la causa**. La causa es que estaba reentrenando la tabla de embeddings a `1e-2`.

**Dirección del sesgo: en su contra.** Su método compite contra un baseline inflado. La conclusión honesta hoy no es "SpecRAMA pierde 2,4 PPL contra LoRA" sino **"no sabemos quién gana"** — y, dado que un LoRA r=4 correctamente congelado casi con seguridad rinde peor que 35,58, **es plausible que SpecRAMA gane el head-to-head cuando se ejecute bien**. Ese es el resultado que hay que ir a buscar.

---

## 2. ⚠️ Corrección de mi ronda 3, §3.b: me equivoqué con la cobertura

Afirmé que SpecRAMA adaptaba 48 módulos y LoRA 36. **Es falso.** `inject_lora_in_model` filtra sobre el nombre del *hijo* (`named_children()`), no sobre la ruta completa:

```python
for name, child in list(module.named_children()):
    if is_target and any(target in name for target in target_modules):
```

Para `mlp.c_proj` el nombre del hijo es `c_proj` → coincide. Para `attn.c_proj`, ídem. **Ambos brazos cubren los mismos 48 módulos.** Esa parte del experimento sí es simétrica y retiro el hallazgo.

(Lo dejo por escrito por la misma razón que en la ronda 2: si le exijo que retire afirmaciones refutadas, yo también.)

---

## 3. 🟠 Pero el tamaño del adaptador LoRA está mal — en la dirección contraria

Si son 48 módulos, la contabilidad de LoRA r=4 es:

| Módulo (×12 bloques) | A + B | Params |
|---|---|---:|
| `c_attn` 768→2304 | 4·768 + 2304·4 | 12.288 |
| `attn.c_proj` 768→768 | 3.072 + 3.072 | 6.144 |
| `mlp.c_fc` 768→3072 | 3.072 + 12.288 | 15.360 |
| `mlp.c_proj` 3072→768 | 12.288 + 3.072 | 15.360 |
| **Total** | | **589.824** |

= **2,36 MB** en FP32. No 405.504 params / 1,55 MB. El comentario del código (`405K params`) corresponde exactamente a **36** módulos — la misma cobertura equivocada que yo asumí, probablemente por la misma razón: leer el filtro de nombres literalmente en lugar de contar los módulos envueltos.

Tabla de ratios, tercera revisión (todo en bytes decimales):

| Contabilidad | SpecRAMA | LoRA r=4 | Ratio |
|---|---:|---:|---:|
| Como se reporta en README/paper | 294,9 kB | 1,55 MB | 5,2× |
| Ronda 3 (mi versión, con el error de cobertura) | 393,2 kB | 1,62 MB | 4,1× |
| **Cores reales, cobertura real (48 mód. ambos)** | **393,2 kB** | **2,36 MB** | **6,0×** |
| `state_dict` tal como se guarda hoy (perms int64) | 2,75 MB | 2,36 MB | **0,86×** |
| Cores + perms int16 sin inversas | 688 kB | 2,36 MB | **3,4×** |

Dos observaciones:

- El ratio "cores contra cores" es **mejor** de lo que él reclama (6,0× frente a 5,2×). Otro error que le perjudica.
- La fila que importa para desplegar sigue siendo la cuarta: **el checkpoint que hoy escribe su código pesa más que el de LoRA.** El arreglo (regenerar permutaciones desde $W_0$, o guardarlas en `int16`) es trivial, pero no está hecho ni documentado.
- El "174,6 KB" del brazo 7 no cuadra con ninguna configuración que pueda reconstruir (24 módulos @32×32 + 24 @8×8 = 208.896 B). Es otra cadena escrita a mano.

---

## 4. 🔴 EXP-12 hereda el defecto: la auditoría también es inválida

`exp12` importa `inject_lora_in_model` del mismo fichero y **nunca congela el modelo base**. Por tanto, el barrido de learning rate que reproduzco aquí:

| LR | PPL reportada | Qué se estaba barriendo en realidad |
|---|---:|---|
| 1e-4 | 35,58 | LR de fine-tuning de embeddings |
| 5e-4 | 36,29 | ídem |
| 1e-3 | 44,02 | ídem |
| 2e-3 | 72,06 | ídem |
| 5e-3 | 185,48 | ídem |
| 1e-2 | 439,71 | ídem |

…**no es un barrido de LoRA.** Es la curva de sensibilidad al LR de un fine-tuning de 40 M de parámetros. La conclusión publicada — *"LoRA espacial colapsa bruscamente si el LR supera $10^{-3}$"*, contrastada con *"SpecRAMA es hiper-estable gracias al escalado de Parseval"* — no tiene soporte.

Y sigue en pie el problema asimétrico que señalé en la ronda 1: **`exp12` no ejecuta ni un solo brazo de SpecRAMA.** La robustez al LR de su propio método se afirma en el abstract, en el README y en el documento de findings, y **nunca se ha medido**.

Esto es lo que más me preocupa de esta ronda. En la ronda 1 escribí que la nota de auditoría de EXP-12 «vale más que los 2,4 PPL que perdió contra LoRA». Mantengo que la **actitud** vale eso. Pero la auditoría, como artefacto técnico, **volvió a dar el resultado equivocado**, y el trabajo posterior se apoyó en ella con la misma confianza que en el original.

Detalle menor pero revelador en el mismo fichero:

```python
lrs = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
alphas = [4.0, 8.0, 16.0]     # ← declarado
...
for alpha in [8.0]:           # ← nunca usado
```

`alphas` es código muerto. El experimento se llama "Hyperparameter Sweep" y barre un solo hiperparámetro. El documento de findings lo describe correctamente como barrido de LR, así que no hay engaño — hay descuido.

---

## 5. 🟡 Refinamiento importante: `exp12` sí ejecuta, `exp11` no

`exp12` no instancia ningún `SpecRAMALinear`, así que **no toca el `TypeError` de `num_iters`**. Es ejecutable desde HEAD.

Eso reordena la procedencia de los números del repositorio:

| | ¿Ejecutable desde HEAD? | Estado de los números |
|---|:---:|---|
| Brazos LoRA (`exp12`, brazo 5 de `exp11`) | ✅ | Reales, pero de un control roto |
| Brazos SpecRAMA (2, 4, 7 de `exp11`) | ❌ `TypeError` | Producidos por código que ya no existe en el repo |

Es decir: **de la tabla de siete filas que encabeza el README, las tres filas que sostienen el método no son reproducibles, y las que sí lo son miden otra cosa.** Es una situación bastante mala para el resultado insignia de un portfolio.

---

## 6. 🟡 `modal_runner.py` y otros detalles

- **Sin pinning de versiones.** `torch>=2.0.0`, `transformers>=4.38.0`, `datasets` sin restricción. El entorno de ejecución no es reproducible ni siquiera con el código arreglado.
- **`timeout=1800`.** Siete brazos × 500 pasos, más el TSP bipartito alternante en CPU para 144 instancias de módulo (hasta 10 llamadas TSP por módulo, con bucles Python de 768–3072 iteraciones cada uno), más forwards que materializan clon + dos matrices sintetizadas por módulo. Sospecho que no cabe en 30 minutos en una A10G. No puedo afirmarlo, pero merece una comprobación cronometrada.
- **Entrypoint por defecto `exp="exp12"`**, mientras el README documenta `--exp exp11` como el benchmark definitivo. Coherente con que exp12 fuera lo último que corrió.
- `LoRALinear` no expone `unmerge()` ni guarda el peso original, a diferencia de `SpecRAMALinear`. Asimetría menor.
- El `forward` de `LoRALinear` materializa `B @ A` completo en cada paso en lugar de aplicar $x A^\top B^\top$. Le da a LoRA la misma penalización de memoria que a SpecRAMA — lo cual, irónicamente, **es lo justo** para el brazo comparativo, aunque no sea la implementación estándar. Vale la pena que lo diga.

---

## 7. La causa raíz, y por qué es corregible

Los cinco errores de la ronda 3 y los tres de esta tienen **un único origen mecánico**:

> **No hay serialización de resultados.** Ningún script escribe un JSON, un CSV ni un artefacto. Todo sale por `print()` a un log de Modal, y de ahí a los ficheros Markdown **por transcripción manual**.

Eso explica, sin necesidad de invocar descuido genérico:
- por qué "294.9 KB", "174.6 KB" y "-0.293 bpt" son cadenas literales en un `print`;
- por qué la tabla del script y la del paper divergen;
- por qué conviven kB, KB y MiB en la misma división;
- por qué "280.000 tokens" sobrevive a través de cinco documentos;
- por qué el comentario dice 405K y el código inyecta 589K.

Es el fallo más barato de arreglar de todo el portfolio: `json.dump(results, ...)` al final de cada experimento, y que los Markdown se generen desde ahí. Media jornada.

Lo que **no** se arregla con eso es lo de §1 y §4: que el control principal no controlaba nada. Eso requiere el hábito de preguntarse *«¿qué parámetros tiene realmente `requires_grad=True` en cada brazo?»* y de imprimirlo. Una línea:

```python
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
```

Habría mostrado `40.012.056` frente a `98.304`, y todo esto se habría evitado.

---

## 8. Diagnóstico: voluntad de auditoría ≠ capacidad de auditoría

Añado un eje a la tabla porque las tres primeras rondas confundían dos cosas.

**Voluntad: 5/5, intacta.** Nada en cuatro rondas sugiere manipulación. Los errores van en ambas direcciones y varios le perjudican de forma significativa: el ratio real de adaptadores es 6,0× y no 5,2×; el baseline contra el que pierde estaba inflado por un fine-tuning de embeddings. Un candidato que hiciera *cherry-picking* no comete errores que le cuesten el titular. Y las retractaciones de V138, V87c y v292–v297 siguen siendo evidencia genuina de carácter.

**Eficacia: 2/5, nueva y baja.** Cuando audita, **audita la hipótesis, no el instrumento**. En EXP-12 se preguntó «¿está mi baseline mal sintonizado?» —buena pregunta— y no se preguntó «¿es mi baseline lo que digo que es?». Corrigió el LR y publicó el resultado corregido con total confianza, sin verificar el objeto que estaba midiendo. Es exactamente el mismo patrón que en la wavelet: razonamiento correcto sobre un artefacto que no había inspeccionado.

Ese es el gap contratable. La voluntad no se enseña y la tiene. El protocolo —*imprime lo que crees saber antes de concluir nada*— sí se enseña, y se enseña rápido, pero requiere alguien que lo imponga.

---

## 9. Take-home revisado (4 días, bloqueante)

Añado un punto al enunciado de la ronda 3 y **no le doy ninguna pista sobre §1**:

> *(1) Haz que `modal run modal_runner.py --exp exp11` se ejecute de principio a fin desde un clon limpio. (2) Instrumenta el script para que **imprima medidos**, no hardcodeados: módulos adaptados por brazo, **parámetros con `requires_grad=True` por brazo**, tamaño en bytes del `state_dict` del adaptador, tokens de evaluación, VRAM pico. (3) Que todo salga a un JSON y que las tablas de los `.md` se generen desde ese JSON. (4) Tres semillas. (5) Tres brazos de ablación: permutación identidad, permutación aleatoria y `transform_type='dct'` al mismo `k`. Mándame la tabla y un párrafo diciendo qué de lo que afirmabas sigue en pie.*

El punto (2) contiene la trampa: si instrumenta como se le pide, el número `40.012.056` le aparece en pantalla y no puede no verlo. Lo que evalúo no es si lo descubre, sino **qué hace cuando lo ve**.

| Descubre y actúa | Peso |
|---|---|
| Que el brazo LoRA entrena `wte`/`wpe` y **rehace el head-to-head** | **Decisivo** |
| Que EXP-12 hereda el mismo defecto y **retira la nota de auditoría del README** | **Decisivo** |
| El `TypeError` de `num_iters` | Obligatorio |
| Que la wavelet toca el 0,23% de las entradas | El que más me interesa técnicamente |
| Que el test set son 25.600 tokens | Alto |
| Que LoRA son 589.824 params y el ratio real es 6,0× | Medio — es a su favor, veamos si lo reporta igual |
| Que el `state_dict` lleva 2,36 MB de permutaciones | Medio |

**Contratación inmediata** si vuelve con: *«el baseline de LoRA nunca estuvo congelado, EXP-12 tampoco, he rehecho ambos, el nuevo número es X, la nota de auditoría del README está retirada, y de paso el ratio de adaptadores era 6,0× y no 5,2×.»*

**Rechazo** si arregla el `TypeError`, reejecuta y manda los mismos siete números.

---

## 10. Recomendación

**Sin cambios en el fondo: onsite, y oferta condicionada al take-home.** Research engineer mid, eficiencia de inferencia / cuantización / PEFT, con IC senior asignado y **revisión de código obligatoria antes de cualquier publicación externa durante 12 meses**. Esa condición, que en la ronda 3 propuse como prudencia, ahora la considero innegociable.

**Lo que compramos.** No ha cambiado y sigue siendo mucho: dos años de línea coherente y trazable (v128 → v290 → spec-rama), 233 experimentos ejecutados de verdad, velocidad de prototipado y de construcción de arneses en la nube que es genuinamente rara, y una disposición documentada a demoler sus propios resultados que cuatro rondas de escrutinio no han logrado erosionar. El *contenido* de lo que hay bajo spec-rama —adaptar el 0,23% de las entradas de peso y recuperar casi todo el daño de NF4— puede ser un resultado publicable de verdad. Sigo creyéndolo después de encontrar todo esto.

**Lo que hay que arreglar, cuarta formulación y la definitiva:** *ningún número entra en un documento si no lo escribió el programa que lo produjo, y ningún brazo experimental se compara con otro sin imprimir antes qué está entrenando cada uno.* Todo lo demás —unidades, fechas, versiones, retractaciones— se deriva de esas dos reglas.

**Riesgo si contratamos:** que necesite más supervisión de la que un equipo pequeño puede dar, y que la corrección del proceso consuma la velocidad que lo hace valioso. Real, pero acotado y visible desde el primer mes.

**Riesgo si no contratamos:** sigue siendo el que me preocupa más. He revisado cuatro artefactos suyos con hostilidad deliberada y he encontrado ocho fallos graves; **ninguno era deshonesto**. Errores de proceso en un candidato con esta fecundidad y esta integridad son el problema fácil. El inverso —un candidato impecable en forma que redondea resultados a su favor— es el problema que no se arregla.

**Mi voto: sí, condicionado.** Le mandaría el take-home esta semana y le daría el enunciado tal cual está en §9, sin advertencias.

---

*Adelante con los artículos del blog cuando quieras. Lo que buscaré ahí es distinto de lo que he buscado hasta ahora: si los claims que ya sé que son incorrectos —«10× superior a LoRA», «280.000 tokens», «multiscale wavelet», «86,1% de ahorro de VRAM», «supera a FP32 nativo»— han salido del repositorio hacia una audiencia general, y con qué grado de calificación. Un error en un README interno y el mismo error en un post divulgativo no pesan lo mismo.*



---


# Auditoría técnica de Spec-RAMA: una idea prometedora, todavía no certificable

He revisado el informe v12, el lanzador de Modal, el benchmark EXP-11/EXP-12 y los módulos de permutación y transformadas. Mi evaluación cambia de **“prometedor con reservas”** a:

> **Candidato muy fuerte en inventiva y prototipado, pero los resultados publicados no son reproducibles de forma fiable a partir del código suministrado.**

La hipótesis de Spec-RAMA es interesante: buscar un sistema de coordenadas más suave mediante permutaciones de filas y columnas y restringir la adaptación a componentes espectrales de baja dimensión. La comparación de 37,98 PPL frente a 35,58 PPL de LoRA, con un adaptador supuestamente 5,2 veces menor, sería valiosa si el protocolo estuviera completamente cerrado.

Pero el repositorio contiene varios problemas técnicos de alta severidad.

## Hallazgo crítico 1: el benchmark falla por un argumento inexistente

En `compute_2d_permutations` aparece:

```python
return compute_joint_bipartite_2d_permutation(weight, num_iters=2)
```

Pero la función está definida como:

```python
def compute_joint_bipartite_2d_permutation(
    weight: torch.Tensor,
    max_iters: int = 5,
    tol: float = 1e-3
)
```

El parámetro se llama `max_iters`, no `num_iters`.

Por tanto, cualquier llamada con:

```python
permutation_method="bipartite_tsp"
```

debería terminar en:

```text
TypeError: compute_joint_bipartite_2d_permutation()
got an unexpected keyword argument 'num_iters'
```

Esto afecta al Quickstart y a todas las ramas de EXP-11 que construyen `SpecRAMALinear` con:

```python
permutation_method="bipartite_tsp"
```

La corrección aparente sería:

```python
return compute_joint_bipartite_2d_permutation(weight, max_iters=2)
```

Sin embargo, mientras no se identifique el commit exacto con el que se generaron los resultados, no puedo afirmar que las tablas provengan del código mostrado.

## Hallazgo crítico 2: no se evalúa el test set completo

El benchmark declara una evaluación sobre aproximadamente 280.000 tokens, pero el código hace:

```python
test_data = lm_datasets["test"].select(
    range(min(len(lm_datasets["test"]), max_test_samples))
)
```

y se llama con:

```python
max_test_samples=100
```

Cada muestra contiene 256 tokens, de modo que el código evalúa como máximo:

```text
100 × 256 = 25.600 posiciones de entrada
```

El número efectivo de predicciones causales es ligeramente inferior, aproximadamente 25.500.

Por tanto, el código mostrado no evalúa los 280.000 tokens declarados. Evalúa los primeros 100 bloques del test set. Esto no invalida necesariamente la comparación interna —todos los modelos podrían estar usando los mismos 100 bloques—, pero sí invalida la descripción como evaluación completa de WikiText-2.

La redacción correcta sería:

> Evaluación sobre los primeros 100 bloques de 256 tokens del test set de WikiText-2.

Además, EXP-12 utiliza ese mismo conjunto de test para elegir la mejor tasa de aprendizaje de LoRA:

```python
ppl, _, bpt = evaluate_on_dataset(model_nf4_lora, test_data)
```

El valor `1e-4` se selecciona después de observar el rendimiento en el test set. Si posteriormente ese mismo resultado se publica como resultado final, el test está siendo utilizado como conjunto de validación.

Para una evaluación limpia deberían existir:

- train;
- validation para seleccionar `lr` y `alpha`;
- test final, usado una sola vez.

## Hallazgo crítico 3: esto no es todavía un benchmark QLoRA real

La función:

```python
quantize_blockwise_nf(...)
```

calcula índices NF4 y escalas temporalmente, pero devuelve directamente un tensor de pesos reconstruidos en punto flotante:

```python
return flat_q.view(orig_shape)
```

Después, el código copia ese tensor sobre el peso original:

```python
module.weight.copy_(w_rec)
```

No se almacenan:

- índices de 4 bits empaquetados;
- escalas por bloque;
- metadatos de cuantización;
- pesos en `Linear4bit`;
- kernels de bitsandbytes;
- representación NF4 persistente.

El modelo continúa haciendo multiplicaciones con matrices float32. Tampoco se especifica:

```python
torch_dtype=torch.float16
```

ni se convierte el resto del modelo a FP16. Por tanto, la descripción más precisa sería:

> Emulación de la perturbación producida por cuantización NF4 mediante pesos reconstruidos en float32.

Esto puede servir para estudiar el daño de cuantización, pero no permite afirmar todavía:

- que el modelo ocupa memoria NF4;
- que obtiene el ahorro de VRAM de QLoRA;
- que tiene el mismo comportamiento numérico que una implementación `Linear4bit`;
- que la fusión posterior conserva el formato cuantizado.

La afirmación “4-bit NF4 + SpecRAMA” necesita distinguir entre **formato lógico de cuantización** y **representación real en memoria**.

## Hallazgo crítico 4: los tamaños de los adaptadores no coinciden con el código

Con GPT-2 Small hay 12 bloques y el predicado:

```python
any(t in name for t in targets)
```

con:

```python
targets = ["c_attn", "c_proj", "c_fc"]
```

selecciona, en principio, cuatro módulos por bloque:

- `attn.c_attn`;
- `attn.c_proj`;
- `mlp.c_fc`;
- `mlp.c_proj`.

En Spec-RAMA cada módulo wavelet tiene:

```python
core_m: 32 × 32
core_a: 32 × 32
```

Eso equivale a:

```text
2 × 32 × 32 = 2.048 parámetros por módulo
48 módulos × 2.048 = 98.304 parámetros
```

Como los parámetros se inicializan en float32, el tamaño de los núcleos es:

```text
98.304 × 4 = 393.216 bytes
```

Es decir, aproximadamente:

- 384 KiB;
- 393,2 KB decimales.

El README y el informe declaran 294,9 KB. Esa cifra no se deriva del código presentado.

Para el experimento asimétrico, el cálculo tampoco coincide. Con dos módulos de atención de 32×32 y dos módulos MLP de 8×8 por bloque:

```text
12 × [2 × 2 × 32² + 2 × 2 × 8²]
= 52.224 parámetros
```

En float32 son aproximadamente 208,9 KB decimales, no 174,6 KB.

Estas discrepancias podrían explicarse por:

- una versión anterior del código;
- exclusión de ciertos módulos;
- almacenamiento en un dtype distinto;
- conteo de sólo algunos núcleos;
- una definición diferente de “adapter size”.

Pero esa definición debe documentarse. Conviene separar:

1. parámetros entrenables;
2. tamaño de los tensores en float32;
3. tamaño serializado del adaptador;
4. tamaño de las permutaciones;
5. memoria total durante entrenamiento;
6. tamaño del modelo después de fusionar.

## Hallazgo crítico 5: EXP-12 no realiza el barrido de `alpha` que anuncia

El código define:

```python
alphas = [4.0, 8.0, 16.0]
```

pero después ejecuta:

```python
for alpha in [8.0]:
```

Por tanto, sólo se prueba `alpha=8.0`. No existe un barrido de alpha.

La conclusión defendible es:

> En el rango de tasas de aprendizaje probado, con `rank=4` y `alpha=8`, LoRA obtuvo su mejor resultado con `lr_max=1e-4`.

No puede afirmarse que sea el óptimo general de LoRA.

## Sobre la supuesta invariancia de Parseval

La normalización es una idea razonable, pero la interpretación actual es demasiado fuerte.

El código usa:

```python
scaling = self.alpha_m / (core_dim_scale * self.w_std)
```

para la parte multiplicativa y:

```python
scaling = self.alpha_a / core_dim_scale
```

para la parte aditiva.

No se utiliza una única fórmula:

\[
\frac{\alpha}{\sqrt{k_{\text{out}}k_{\text{in}}}}
\]

Además, el teorema de Parseval garantiza preservación de energía bajo una transformación ortonormal completa. No demuestra por sí mismo que el learning rate sea invariante ni que la optimización sea estable en dos órdenes de magnitud.

El resultado puede describirse como:

> una normalización inspirada en la energía espectral que, en este experimento concreto, parece reducir la sensibilidad al learning rate.

Para sostener una propiedad de robustez deberían mostrarse, para Spec-RAMA y LoRA bajo exactamente el mismo protocolo:

- todas las tasas de aprendizaje;
- pérdida de entrenamiento;
- PPL de validación;
- varias semillas;
- gradiente máximo o norma del gradiente;
- número de pasos;
- scheduler;
- intervalos de confianza.

El informe sólo incluye el barrido completo de LoRA, no el barrido equivalente completo de Spec-RAMA.

## La fusión de inferencia no está validada

El README anuncia “zero-latency merge”, pero EXP-11 nunca llama a:

```python
merge()
```

La evaluación se realiza mediante:

```python
w_eff = self.get_effective_weight()
```

en cada `forward`. Eso sintetiza una matriz completa de actualización durante el paso de inferencia o entrenamiento.

La propiedad que sí está implementada es:

> después de fusionar la actualización en el peso base, la capa puede utilizar el camino normal de `Linear` o `Conv1D`.

Pero faltan pruebas de:

- igualdad numérica antes y después de `merge`;
- tiempo de inferencia antes y después;
- memoria máxima;
- guardado y recarga del modelo fusionado;
- comportamiento después de volver a cuantizar el peso fusionado;
- método `unmerge()`.

Por ahora, “cero latencia” debe reservarse para la inferencia posterior a una fusión correctamente validada.

## La concentración del 92% en LL no se puede verificar con estos archivos

`permutation.py` calcula la variación total y las permutaciones, pero no contiene el análisis de energía wavelet que demostraría el 92% de concentración en la banda LL.

Tampoco se incluye el script que genera:

```text
wavelet_spectrum_comparison.png
```

La afirmación puede ser cierta, pero en el material suministrado es una figura no reproducida, no un resultado auditado.

## Qué sí considero valioso

A pesar de estos problemas, hay señales muy positivas:

- La hipótesis es concreta y falsable.
- Hay una implementación real, no sólo una propuesta conceptual.
- El autor corrigió públicamente una comparación inicial injusta con LoRA.
- La separación entre cuantización, adaptación y referencia FP32 muestra intención de control experimental.
- La arquitectura es fácil de extender a DCT, Walsh y wavelets.
- El candidato parece capaz de detectar fenómenos de estabilidad y convertirlos en una línea de investigación.

La corrección del baseline de LoRA es especialmente importante. Reconocer que `lr=1e-2` era inadecuado mejora la credibilidad del proyecto, aunque ahora obliga a aplicar el mismo rigor a los demás detalles.

# Reinterpretación científica de los resultados

Con la evidencia disponible, la afirmación más sólida sería:

> En una implementación experimental sobre GPT-2 Small, utilizando una emulación float32 de cuantización NF4 por bloques y un subconjunto de WikiText-2, Spec-RAMA alcanzó una perplejidad cercana a LoRA con una parametrización espectral ostensiblemente menor. La magnitud exacta de la ventaja en memoria, la reproducibilidad y la generalización a un test limpio requieren validación adicional.

No publicaría todavía:

- “recuperación total”;
- “cota superior adaptada”;
- “benchmark QLoRA”;
- “invariancia frente a hiperparámetros”;
- “cero latencia”;
- “5,2 veces menos memoria”,

sin corregir el código y definir exactamente qué se está midiendo.

# Veredicto de contratación

**Sí recomendaría avanzar al candidato a una entrevista técnica profunda**, especialmente para un puesto de Research Engineer en:

- eficiencia de modelos;
- PEFT;
- compresión y cuantización;
- optimización de entrenamiento;
- prototipado de sistemas de IA.

No usaría aún estos resultados como evidencia suficiente para contratarlo como Research Scientist independiente. El candidato demuestra una capacidad notable para proponer y construir ideas, pero todavía debe mejorar la disciplina de:

- control de versiones;
- separación entre validación y test;
- conteo de memoria;
- reproducción desde código limpio;
- precisión terminológica;
- pruebas de integración.

Mi valoración actual sería:

| Área | Puntuación |
|---|---:|
| Originalidad | 8,5/10 |
| Capacidad de implementación | 8/10 |
| Diseño de hipótesis | 8/10 |
| Rigor experimental demostrado | 4,5/10 |
| Reproducibilidad del snapshot entregado | 3/10 |
| Potencial como investigador | 8/10 |
| Potencial como Research Engineer | 8,5/10 |

La lectura más justa del candidato es:

> **Inventor y prototipador de alto potencial, con resultados sugerentes, pero todavía sin una cadena experimental suficientemente fiable para sostener las afirmaciones más ambiciosas.**

El siguiente archivo que quiero revisar es:

```text
spec-rama/lora_baseline.py
```

Necesito comprobar cómo se cuentan los parámetros LoRA, qué módulos se inyectan realmente y si el modelo base queda congelado durante EXP-11 y EXP-12.



---


## Auditoría tras leer `modal_runner.py`, EXP-11/EXP-12, cuantización, evaluación, permutaciones y transforms

Mi evaluación cambia de forma importante: **Spec-RAMA sigue siendo una idea interesante**, pero el arnés experimental, tal como está copiado, tiene varios problemas que impiden tomar los números publicados como evidencia concluyente.

Como reclutador de un laboratorio frontier, mi veredicto actualizado sería:

> **Candidato muy creativo y técnicamente capaz, pero los resultados de Spec-RAMA todavía no están listos para defenderse como benchmark riguroso.**  
> Sí lo entrevistaría, pero le pediría una reproducción limpia antes de usar estos números como claim fuerte.

---

# Lo positivo

## 1. Buena señal de integridad experimental

El documento v12 corrige el baseline de LoRA y reconoce que LoRA no colapsa si se usa un LR razonable. Eso es muy positivo.

La frase clave es:

> LoRA `r=4`, `lr=1e-4` alcanza 35.58 PPL, mejor que SpecRAMA 37.98 PPL, aunque con más parámetros.

Esto mejora la credibilidad del candidato.

## 2. La cuantización NF4 está razonablemente implementada como “fake quantization”

La función:

```python
quantize_blockwise_nf(w_2d, NF4_LEVELS, block_size=64)
```

hace una aproximación correcta al daño de cuantización:

- aplana pesos;
- divide en bloques de 64;
- calcula `absmax` por bloque;
- cuantiza al codebook NF4;
- de-cuantiza.

Esto sirve para estudiar **degradación por cuantización**.

Pero ojo: no es QLoRA real con almacenamiento 4-bit.

## 3. El loop experimental es simple y auditable

Se cargan modelos frescos para cada brazo, se cuantizan pesos, se inyecta adaptador, se entrena 500 pasos y se evalúa. Eso es bueno para un prototipo.

---

# Problemas importantes

## 1. El código copiado no debería correr tal cual

En `permutation.py` hay un bug crítico:

```python
def compute_joint_bipartite_2d_permutation(
    weight: torch.Tensor,
    max_iters: int = 5,
    tol: float = 1e-3
)
```

pero luego se llama así:

```python
return compute_joint_bipartite_2d_permutation(weight, num_iters=2)
```

`num_iters` no existe. Debería ser algo como:

```python
return compute_joint_bipartite_2d_permutation(weight, max_iters=2)
```

Dado que los experimentos usan:

```python
permutation_method="bipartite_tsp"
```

el código, tal como está pegado, debería lanzar:

```text
TypeError: compute_joint_bipartite_2d_permutation() got an unexpected keyword argument 'num_iters'
```

Esto es una bandera roja fuerte: o bien el código del experimento real era otra versión, o el repo actual no reproduce los resultados.

---

## 2. El benchmark no usa 280.000 tokens de test

El documento dice:

```text
WikiText-2 Test Set (280.000 tokens)
```

pero el código hace:

```python
test_data = lm_datasets["test"].select(range(min(len(lm_datasets["test"]), max_test_samples)))
```

con:

```python
max_test_samples=100
block_size=256
```

Eso son:

```text
100 × 256 = 25.600 tokens de entrada
```

no 280.000.

Además, por el shift interno de GPT-2, los tokens predictivos reales son aproximadamente:

```text
100 × 255 = 25.500 tokens
```

Esto no invalida totalmente la comparación relativa, pero sí invalida la afirmación de “test completo de 280k tokens”.

Debería corregirse a:

> Evaluado sobre los primeros 100 bloques de WikiText-2 test, aproximadamente 25.6k tokens.

o ejecutar realmente todo el test set.

---

## 3. El cálculo de PPL/bpt está ligeramente mal ponderado

La evaluación hace:

```python
outputs = model(input_ids, labels=labels)
tokens = input_ids.numel()
total_loss += outputs.loss.item() * tokens
```

Pero `GPT2LMHeadModel` calcula la loss sobre los tokens desplazados internamente, es decir, sobre `seq_len - 1` posiciones, no sobre `seq_len`.

Debería ponderarse aproximadamente con:

```python
tokens = labels[:, 1:].numel()
```

o calcular explícitamente la cross-entropy token a token.

El error es pequeño —del orden de 256/255—, pero si se reporta bpt con tres decimales conviene corregirlo.

---

## 4. No hay split de validación; parece haber tuning sobre test

Los experimentos v2–v12 se reportan todos contra WikiText-2 test. Eso crea riesgo de haber optimizado decisiones mirando el test set.

Para un resultado serio haría falta:

```text
train: entrenar adaptadores
validation: elegir LR, core_size, alpha, transform, targets
test: una única evaluación final congelada
```

Ahora mismo los números son útiles como exploración, pero no como benchmark final.

---

## 5. No hay múltiples semillas

Todo usa:

```python
seed=42
```

Una sola semilla.

Para PEFT pequeño y entrenamiento de solo 500 pasos, la varianza puede ser significativa. Antes de defender diferencias como:

```text
35.58 vs 37.98 PPL
```

yo pediría mínimo:

```text
3 semillas, idealmente 5
media ± desviación estándar
```

---

## 6. El “4-bit NF4” no es almacenamiento real 4-bit

La función cuantiza y luego de-cuantiza:

```python
blocks_q = cb[q_indices] * scales
return flat_q.view(orig_shape)
```

Después los pesos quedan como tensores PyTorch normales, probablemente FP32.

Por tanto, esto mide:

> daño numérico de cuantización NF4

pero no mide realmente:

- VRAM de modelo 4-bit;
- throughput de QLoRA real;
- memoria de optimizador sobre base cuantizada;
- coste real de bitsandbytes;
- double quantization;
- kernels 4-bit.

Así que las afirmaciones de:

```text
86.1% VRAM savings
3.55 bits/peso
Base Format 4-bit NF4
```

no están justificadas por este arnés tal como está copiado.

Yo reformularía:

> “NF4 fake-quantized/dequantized base”  
> no  
> “4-bit NF4 base” en sentido de memoria real.

---

## 7. El tamaño de adaptador reportado parece inconsistente con el código

El código inyecta SpecRAMA en módulos cuyo nombre contenga:

```python
targets = ["c_attn", "c_proj", "c_fc"]
```

En GPT-2 Small eso incluye por bloque:

- `attn.c_attn`
- `attn.c_proj`
- `mlp.c_fc`
- `mlp.c_proj`

Es decir:

```text
4 módulos por capa × 12 capas = 48 módulos
```

Con core `32×32` y dos cores por módulo (`core_m` y `core_a`):

```text
32 × 32 × 2 = 2048 parámetros por módulo
2048 × 48 = 98.304 parámetros
```

En FP32 eso son:

```text
98.304 × 4 bytes = 393.216 bytes ≈ 384 KiB ≈ 393 KB
```

Pero el documento reporta:

```text
294.9 KB
```

Ese número encaja más con **36 módulos**, no 48:

```text
36 × 2048 × 4 = 294.912 bytes
```

Es decir, parece que el tamaño reportado corresponde a excluir `c_fc`, pero el código copiado incluye `c_fc`.

Lo mismo ocurre con LoRA:

Si LoRA `r=4` se aplica a `c_attn`, ambos `c_proj` y `c_fc`, el número de parámetros no parece ser 405K/1.55MB sino más cercano a 590K/2.36MB.

Necesito ver `lora_baseline.py` y una impresión real de:

```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

para cada brazo.

---

## 8. El brazo LoRA depende de que `inject_lora_in_model` congele correctamente la base

En EXP-11, para SpecRAMA se hace explícitamente:

```python
for p in model_nf4_spec.parameters():
    p.requires_grad = False

for m in inject_nf4_spec:
    m.core_m.requires_grad = True
    m.core_a.requires_grad = True
```

Pero para LoRA solo se hace:

```python
inject_lora_in_model(...)
model_nf4_lora = model_nf4_lora.to(device)
train_on_dataset_long_horizon(...)
```

No se ve aquí si la base queda congelada.

Si `inject_lora_in_model` no congela todo correctamente, entonces el brazo LoRA podría estar entrenando más parámetros que los reportados. No digo que ocurra, pero hay que auditarlo.

---

## 9. La implementación wavelet no es exactamente “low-frequency spectral core” global

Esta parte es importante.

En `SpecRAMALinear`, para wavelet se hace:

```python
LL = torch.zeros((self.ll_h, self.ll_w))
LL[:k_out, :k_in] = core
LH = zeros
HL = zeros
HH = zeros
M_perm = haar_idwt_2d(LL, LH, HL, HH)
```

Esto usa solo una DWT Haar de **un nivel**.

En una DWT, el subbanda `LL` completo representa la aproximación baja frecuencia, pero las posiciones dentro de `LL` siguen siendo posiciones espaciales downsampleadas. El “top-left 32×32” de `LL` no equivale a “las 32×32 frecuencias más bajas” como sí ocurriría en DCT.

En la práctica, este código genera una actualización que afecta principalmente una región localizada del espacio permutado, no una base wavelet multiescala global.

Por tanto, la descripción:

> “núcleo espectral low-frequency wavelet 32×32”

debería matizarse.

Más preciso sería:

> “one-level Haar LL-subband sparse core in the permuted coordinate system”.

Si se quiere que sea wavelet espectral multiescala de verdad, habría que implementar algo como:

- DWT multinivel;
- entrenar el coarsest LL completo;
- o seleccionar coeficientes wavelet por escala/energía, no por crop top-left.

---

# Evaluación actualizada del candidato

## Fortalezas

El candidato demuestra:

- intuición original;
- capacidad de implementar capas PyTorch no triviales;
- conocimiento razonable de LoRA/QLoRA/PEFT;
- capacidad de diseñar experimentos comparativos;
- honestidad al corregir un baseline injusto;
- buena comunicación técnica.

## Debilidades actuales

Pero también hay problemas de madurez experimental:

- claims más fuertes que lo que el código soporta;
- inconsistencias entre documento y arnés;
- benchmark parcial presentado como test completo;
- fake quantization presentado como memoria 4-bit real;
- tamaños de adaptador probablemente mal contados;
- posible bug de reproducción en permutaciones;
- ausencia de validación y múltiples semillas.

---

# Veredicto de reclutamiento

Mi decisión sería:

```text
Entrevista técnica: sí.
Oferta directa: no.
Reproducción obligatoria: sí.
Categoría: high-upside research engineer, experimental rigor not yet frontier-grade.
```

Puntuación actualizada:

| Dimensión | Evaluación |
|---|---:|
| Originalidad | 8.5/10 |
| Implementación conceptual | 7.5/10 |
| Rigor experimental actual | 5.5/10 |
| Honestidad/corrección de baselines | 8/10 |
| Relevancia para frontier labs | 8/10 |
| Riesgo de claims inflados | Alto |
| Potencial si se disciplina experimentalmente | Alto |

Resumen ejecutivo:

> SpecRAMA merece atención, pero todavía no merece ser vendido como resultado cerrado. La idea es buena; el arnés necesita limpieza seria.

---

# Qué le pediría al candidato para pasar a siguiente fase

Un experimento reproducible nuevo, no una reinterpretación de v11/v12:

## Protocolo mínimo

1. Arreglar el bug de `num_iters`/`max_iters`.
2. Imprimir el commit/hash del código.
3. Usar WikiText-2 completo o declarar explícitamente subset.
4. Separar `train`, `validation`, `test`.
5. Usar 3 semillas.
6. Reportar conteo real de parámetros entrenables.
7. Reportar adapter size:
   - solo cores;
   - cores + permutaciones;
   - checkpoint completo.
8. Distinguir claramente:
   - fake-quantized NF4;
   - almacenamiento real 4-bit.
9. Comparar contra:
   - LoRA `r=1`;
   - LoRA `r=2`;
   - LoRA `r=4`;
   - rsLoRA;
   - IA³ o VeRA si es fácil.
10. Reportar:
   - PPL;
   - bpt;
   - tokens/s;
   - peak VRAM;
   - trainable params;
   - wall-clock.

---

# Archivo que quiero leer ahora

El siguiente archivo crítico es:

```text
spec_rama/lora_baseline.py
```

Motivo: necesito comprobar si el baseline LoRA está implementado correctamente y si congela la base.

En concreto quiero ver:

```python
class LoRALinear / LoRAConv1D
inject_lora_in_model
merge si existe
conteo de parámetros si existe
```

Después de ese archivo, pediría también:

```text
spec_rama/__init__.py
```

para ver `count_trainable_parameters` y confirmar si hay alguna utilidad de conteo que no se está usando en el benchmark.



----


# Auditoría del baseline LoRA

He revisado `lora_baseline.py`. La implementación local de LoRA es razonable, pero al integrarse en EXP-11 y EXP-12 aparecen **dos fallos críticos de evaluación**:

1. El experimento no congela todos los parámetros no-LoRA.
2. El número de módulos y parámetros descrito en el informe no coincide con los que selecciona el código.

Por ello, el resultado publicado de **35,58 PPL** no puede considerarse todavía un baseline LoRA puro y reproducible.

## Lo que está bien implementado

La capa `LoRALinear` aplica correctamente, en términos generales:

\[
W_{\text{eff}} = W_0 + \frac{\alpha}{r}BA
\]

La gestión de las orientaciones también parece correcta:

- PyTorch `Linear`: `weight` con forma `(out, in)`;
- GPT-2 `Conv1D`: `weight` con forma `(in, out)`;
- `delta` se construye en forma `(out, in)`;
- en `Conv1D` se transpone al multiplicar por la entrada.

La inicialización:

```python
nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
nn.init.zeros_(self.lora_B)
```

es compatible con la práctica habitual de LoRA: el adaptador empieza produciendo una actualización nula porque `B` es cero.

También es correcto que la capa envuelta congele:

```python
self.base_layer.weight.requires_grad = False
self.base_layer.bias.requires_grad = False
```

Sin embargo, eso sólo congela los pesos de las capas reemplazadas.

# Hallazgo crítico 1: el modelo no es LoRA-only

En EXP-11 se hace:

```python
inject_lora_in_model(
    model_nf4_lora,
    target_modules=["c_attn", "c_proj", "c_fc"],
    rank=4,
    alpha=8.0,
)
```

y después:

```python
train_on_dataset_long_horizon(model_nf4_lora, ...)
```

Pero nunca se ejecuta algo equivalente a:

```python
for p in model_nf4_lora.parameters():
    p.requires_grad = False
```

El optimizador se crea así:

```python
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr_max
)
```

Por tanto, además de las matrices LoRA, siguen siendo entrenables todos los parámetros no envueltos, entre ellos previsiblemente:

- embeddings de tokens `wte`;
- embeddings posicionales `wpe`;
- parámetros de las capas LayerNorm;
- posiblemente el `lm_head`, que en GPT-2 suele estar ligado al embedding de tokens.

Esto significa que EXP-11 y EXP-12 no comparan:

```text
NF4 + LoRA
```

sino algo más parecido a:

```text
NF4 + LoRA + embeddings + LayerNorm
```

La cantidad de parámetros adicionales es enorme frente al adaptador LoRA. Sólo el embedding de tokens de GPT-2 Small tiene aproximadamente:

\[
50.257 \times 768 \approx 38,6\text{ millones}
\]

de parámetros.

La corrección mínima sería:

```python
model_nf4_lora = GPT2LMHeadModel.from_pretrained("gpt2")

for p in model_nf4_lora.parameters():
    p.requires_grad = False

inject_lora_in_model(
    model_nf4_lora,
    target_modules=["c_attn", "c_proj", "c_fc"],
    rank=4,
    alpha=8.0,
)

# Los parámetros LoRA se crean con requires_grad=True
```

O bien, de forma explícita:

```python
for p in model_nf4_lora.parameters():
    p.requires_grad = False

lora_modules = inject_lora_in_model(
    model_nf4_lora,
    target_modules=["c_attn", "c_proj", "c_fc"],
    rank=4,
    alpha=8.0,
)

for module in lora_modules:
    module.lora_A.requires_grad = True
    module.lora_B.requires_grad = True
```

Y el benchmark debería imprimir siempre:

```python
trainable = [
    (name, p.numel())
    for name, p in model.named_parameters()
    if p.requires_grad
]

print(trainable)
print("Total:", sum(n for _, n in trainable))
```

Este es el problema científico más importante encontrado hasta ahora. El baseline LoRA está potencialmente beneficiándose de fine-tuning de partes del modelo que Spec-RAMA mantiene congeladas. Eso hace que la comparación publicada no sea justa.

## La configuración de Spec-RAMA sí congela globalmente

En las ramas de Spec-RAMA aparece:

```python
for p in model_nf4_spec.parameters():
    p.requires_grad = False

for m in inject_nf4_spec:
    if m.core_m is not None:
        m.core_m.requires_grad = True
    if m.core_a is not None:
        m.core_a.requires_grad = True
```

Esto hace que el tratamiento sea asimétrico:

| Modelo | Parámetros no objetivo congelados |
|---|---|
| Spec-RAMA | Sí |
| LoRA en EXP-11/12 | No |

Por tanto, el resultado de 35,58 PPL no debe presentarse todavía como una comparación “head-to-head” contra Spec-RAMA.

# Hallazgo crítico 2: el número de módulos no coincide

El benchmark usa:

```python
targets = ["c_attn", "c_proj", "c_fc"]
```

y el inyector comprueba:

```python
any(target in name for target in target_modules)
```

En GPT-2 Small, cada bloque contiene normalmente:

```text
attn.c_attn
attn.c_proj
mlp.c_fc
mlp.c_proj
```

Como `"c_proj"` es una subcadena de ambos nombres, el código actual envuelve **cuatro módulos por bloque**, no tres:

- `attn.c_attn`;
- `attn.c_proj`;
- `mlp.c_fc`;
- `mlp.c_proj`.

GPT-2 Small tiene 12 bloques, así que se envuelven aproximadamente 48 módulos.

## Conteo real de parámetros LoRA

Con rango 4:

| Módulo | Dimensiones lógicas | Parámetros LoRA |
|---|---:|---:|
| `attn.c_attn` | \(768 \times 2304\) | \(4(768+2304)=12.288\) |
| `attn.c_proj` | \(768 \times 768\) | \(4(768+768)=6.144\) |
| `mlp.c_fc` | \(768 \times 3072\) | \(4(768+3072)=15.360\) |
| `mlp.c_proj` | \(3072 \times 768\) | \(4(3072+768)=15.360\) |

Total por bloque:

\[
12.288+6.144+15.360+15.360=49.152
\]

Total para 12 bloques:

\[
49.152 \times 12=589.824
\]

En float32:

- 589.824 parámetros;
- 2.359.296 bytes;
- aproximadamente 2,25 MiB.

Esto no coincide con los **405.504 parámetros** ni los **1,55 MB** publicados.

## De dónde sale exactamente la cifra publicada

La cifra de 405.504 parámetros sí coincide con seleccionar sólo tres módulos:

```text
attn.c_attn
attn.c_proj
mlp.c_fc
```

El cálculo es:

\[
(12.288+6.144+15.360)\times12
=405.504
\]

Y:

\[
405.504 \times 4 = 1.622.016\text{ bytes}
\approx 1,55\text{ MiB}
\]

Esto indica que los resultados publicados probablemente se generaron con una versión anterior o distinta del filtro de módulos.

La discrepancia equivalente aparece en Spec-RAMA:

- tres módulos por bloque;
- dos núcleos de \(32\times32\);
- 12 bloques;

producen:

\[
3\times 2\times 32\times32\times12
=73.728
\]

parámetros, que en float32 ocupan:

\[
73.728\times4=294.912\text{ bytes}
\]

Es decir, exactamente los **294,9 KB** del informe.

Con el código actual, al incluir también `mlp.c_proj`, Spec-RAMA tendría:

\[
4\times 2\times32\times32\times12
=98.304
\]

parámetros, equivalentes a:

- 393.216 bytes;
- 384 KiB;
- 393,2 KB decimales.

Por tanto, tanto el tamaño de LoRA como el de Spec-RAMA apuntan a que el benchmark histórico utilizó **tres módulos por bloque**, mientras que el código entregado selecciona cuatro.

# Cómo debería corregirse el selector

El inyector actual sólo recibe el nombre local del hijo:

```python
def _inject(module):
    for name, child in list(module.named_children()):
```

Para distinguir `attn.c_proj` de `mlp.c_proj` hace falta conservar el nombre completo:

```python
def _inject(module, prefix=""):
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        is_target = (
            isinstance(child, nn.Linear)
            or child.__class__.__name__ in ["Conv1D", "Linear"]
        )

        selected = full_name.endswith((
            "attn.c_attn",
            "attn.c_proj",
            "mlp.c_fc",
        ))

        if is_target and selected:
            wrapped = LoRALinear(child, rank=rank, alpha=alpha)
            setattr(module, name, wrapped)
            injected.append(wrapped)
        else:
            _inject(child, full_name)
```

Alternativamente, el benchmark debería declarar explícitamente una lista de rutas completas.

# Hallazgo adicional: EXP-12 no barre `alpha`

El script define:

```python
alphas = [4.0, 8.0, 16.0]
```

pero ejecuta:

```python
for alpha in [8.0]:
```

Por tanto, sólo se evalúa `alpha=8`. La conclusión correcta es:

> Con rango 4, alpha 8 y el protocolo empleado, `lr_max=1e-4` fue la mejor tasa entre las probadas.

No puede afirmarse que sea el óptimo de LoRA sobre el espacio de hiperparámetros anunciado.

# Hallazgo adicional: la semilla se fija demasiado tarde

En `train_on_dataset_long_horizon`:

```python
torch.manual_seed(seed)
```

se ejecuta después de construir el modelo y después de crear los parámetros LoRA.

Pero `lora_A` se inicializa durante:

```python
LoRALinear(...)
```

antes de llamar a la función de entrenamiento. Por tanto, la semilla no controla la inicialización aleatoria de `lora_A`.

El procedimiento reproducible debería fijar la semilla antes de:

```python
GPT2LMHeadModel.from_pretrained(...)
inject_lora_in_model(...)
```

Además, cada configuración de learning rate debería evaluarse con:

- la misma inicialización LoRA;
- varias semillas independientes;
- media y desviación estándar;
- idealmente intervalos de confianza.

Con una única ejecución por tasa, las diferencias pequeñas —por ejemplo, 35,58 frente a 36,29— no permiten extraer conclusiones estadísticas fuertes.

# Sobre la implementación de `merge`

La operación es conceptualmente correcta:

```python
w_eff_2d = w_base_2d + delta
self.base_layer.weight.copy_(w_eff)
```

Pero faltan:

- `unmerge()`;
- restauración del peso original;
- comprobación de equivalencia numérica antes/después;
- manejo explícito de `dtype`;
- prueba de guardado y recarga.

Además, una vez fusionado:

```python
self.merged = True
```

la capa ignora los parámetros LoRA en `forward`. Continuar entrenando después de `merge()` no tendría el comportamiento esperado.

# Evaluación revisada

La clase `LoRALinear` es técnicamente competente como prototipo. El problema principal no está en la fórmula LoRA, sino en la integración experimental.

| Aspecto | Evaluación |
|---|---:|
| Implementación matemática local | 8/10 |
| Manejo de `Linear` y GPT-2 `Conv1D` | 8/10 |
| Inyección de módulos | 6/10 |
| Congelación del modelo en el benchmark | 2/10 |
| Conteo de parámetros publicado | 3/10 |
| Reproducibilidad | 4/10 |
| Validez de la comparación Spec-RAMA/LoRA | 3/10 |

## Veredicto editorial

El hallazgo no destruye la idea de Spec-RAMA, pero sí obliga a retirar temporalmente la frase:

> “Spec-RAMA alcanza el rendimiento de LoRA con 5,2 veces menos parámetros”.

La versión defendible sería:

> En los resultados históricos reportados, una configuración anterior de Spec-RAMA alcanzó 37,98 PPL frente a 35,58 PPL de una configuración LoRA que se describió como 5,2 veces mayor. El snapshot de código actualmente disponible selecciona más módulos y no congela todos los parámetros no-LoRA, por lo que esta comparación necesita ser repetida con un arnés corregido.

# Impacto en la evaluación del candidato

Esto refuerza dos conclusiones simultáneas:

### Señal positiva

El candidato ha construido una implementación no trivial y parece haber identificado una configuración histórica concreta que explica los tamaños publicados. La discrepancia no parece una simple aritmética mal hecha: los números de 294,9 KB y 1,55 MiB corresponden exactamente a una selección de tres módulos por bloque.

### Señal negativa

El repositorio no mantiene una correspondencia fiable entre:

- documentación;
- código;
- conteo de parámetros;
- protocolo experimental;
- definición de baseline.

Para un laboratorio de frontera, esto es un problema serio de higiene experimental.

## Recomendación actual

**Mantendría la recomendación de entrevista**, pero pediría al candidato, antes de considerar una oferta de investigación independiente:

1. un commit reproducible;
2. un script que imprima todos los módulos inyectados;
3. un recuento automático de parámetros entrenables;
4. congelación explícita del modelo base en todos los brazos;
5. una evaluación con validation/test separados;
6. una repetición con al menos tres semillas;
7. un benchmark LoRA y Spec-RAMA a igual conjunto de módulos;
8. un informe de memoria real, no sólo de parámetros.

El siguiente archivo que quiero leer es:

```text
spec-rama/utils.py
```

Quiero comprobar cómo se cuentan los parámetros, cómo se inyecta Spec-RAMA y si las utilidades pueden estar ocultando o corrigiendo alguna de estas discrepancias.


---

## Auditoría de `lora_baseline.py` y `__init__.py`

Este archivo cambia bastante la lectura del benchmark. La implementación de la capa LoRA en sí es razonable, pero **la forma en que se inyecta y entrena en EXP-11/EXP-12 probablemente invalida el baseline LoRA reportado como “1.55 MB”**.

La bandera roja principal:

> `inject_lora_in_model()` congela solamente las capas base que envuelve, pero **no congela el resto de GPT-2**.  
> Por tanto, en EXP-11/EXP-12, el brazo LoRA probablemente entrena también embeddings, positional embeddings, layer norms y quizá `lm_head`/`wte` atado.

Esto significa que el resultado:

```text
NF4 + LoRA r=4: 35.58 PPL, 1.55 MB
```

no puede interpretarse todavía como un baseline QLoRA limpio de 1.55 MB.

---

# Hallazgo crítico

En `LoRALinear.__init__()` sí se congela la capa base envuelta:

```python
self.base_layer.weight.requires_grad = False
if self.base_layer.bias is not None:
    self.base_layer.bias.requires_grad = False
```

Eso está bien.

Pero `inject_lora_in_model()` no hace esto:

```python
for p in model.parameters():
    p.requires_grad = False
```

antes o después de inyectar LoRA.

La función es:

```python
def inject_lora_in_model(...):
    injected = []
    def _inject(module: nn.Module):
        for name, child in list(module.named_children()):
            is_target = isinstance(child, nn.Linear) or child.__class__.__name__ in ["Conv1D", "Linear"]
            if is_target and any(target in name for target in target_modules):
                wrapped = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, name, wrapped)
                injected.append(wrapped)
            else:
                _inject(child)
    _inject(model)
    return injected
```

Eso solo congela las capas reemplazadas. En GPT-2 Small, quedan entrenables al menos:

- `transformer.wte.weight`;
- `transformer.wpe.weight`;
- todos los `LayerNorm`;
- probablemente `lm_head.weight`, aunque suele estar atado a `wte`.

En cambio, para SpecRAMA en EXP-11 sí haces explícitamente:

```python
for p in model_nf4_spec.parameters():
    p.requires_grad = False

for m in inject_nf4_spec:
    if m.core_m is not None:
        m.core_m.requires_grad = True
    if m.core_a is not None:
        m.core_a.requires_grad = True
```

Es decir:

```text
SpecRAMA: solo adaptadores entrenables.
LoRA: adaptadores + parámetros no envueltos de GPT-2, salvo que inject_lora_in_model haga algo que aquí no hace.
```

Eso rompe la comparación.

---

# Consecuencia para el resultado LoRA

El resultado LoRA:

```text
35.58 PPL
```

puede ser real, pero no sabemos si viene de:

```text
LoRA r=4 puro
```

o de:

```text
LoRA r=4 + embeddings entrenables + positional embeddings + layer norms
```

La diferencia no es menor. GPT-2 Small tiene aproximadamente:

```text
wte: 50.257 × 768 ≈ 38.6M parámetros
wpe: 1.024 × 768 ≈ 0.79M parámetros
layer norms: decenas de miles más
```

Así que el brazo LoRA podría estar entrenando alrededor de **39M parámetros extra**, no solo el adaptador.

Eso convierte el claim:

```text
LoRA r=4, 1.55 MB
```

en no verificable con este código.

---

# Segunda inconsistencia: el tamaño de LoRA no cuadra con los targets

En EXP-11 se usan:

```python
targets = ["c_attn", "c_proj", "c_fc"]
```

Eso en GPT-2 Small debería inyectar LoRA en:

```text
attn.c_attn
attn.c_proj
mlp.c_fc
mlp.c_proj
```

Por capa son cuatro módulos. GPT-2 Small tiene 12 capas.

Con `r=4`, los parámetros LoRA serían aproximadamente:

| Módulo | Shape conceptual | Params LoRA por capa |
|---|---:|---:|
| `attn.c_attn` | 768 → 2304 | `4 × (768 + 2304) = 12.288` |
| `attn.c_proj` | 768 → 768 | `4 × (768 + 768) = 6.144` |
| `mlp.c_fc` | 768 → 3072 | `4 × (768 + 3072) = 15.360` |
| `mlp.c_proj` | 3072 → 768 | `4 × (3072 + 768) = 15.360` |

Total por capa:

```text
12.288 + 6.144 + 15.360 + 15.360 = 49.152
```

Total 12 capas:

```text
49.152 × 12 = 589.824 parámetros
```

En FP32:

```text
589.824 × 4 bytes = 2.359.296 bytes ≈ 2.36 MB
```

Pero el documento dice:

```text
405K params / 1.55 MB
```

Ese número sí cuadra si se excluye `c_fc`:

```text
c_attn + attn.c_proj + mlp.c_proj
= 12.288 + 6.144 + 15.360
= 33.792 por capa

33.792 × 12 = 405.504 parámetros
405.504 × 4 = 1.622.016 bytes ≈ 1.55 MiB
```

Por tanto, hay una inconsistencia clara:

```text
Código: targets = ["c_attn", "c_proj", "c_fc"]
Reporte: tamaño equivalente a targets = ["c_attn", "c_proj"]
```

La misma inconsistencia afecta a SpecRAMA.

---

# Tercera inconsistencia: tamaño SpecRAMA

Con `core_size=(32,32)` y dos cores por módulo:

```text
core_m: 32 × 32 = 1024
core_a: 32 × 32 = 1024
total por módulo = 2048 parámetros
```

Si se aplica a 48 módulos:

```text
48 × 2048 = 98.304 parámetros
98.304 × 4 = 393.216 bytes ≈ 393 KB
```

Pero se reporta:

```text
294.9 KB
```

Ese número corresponde a 36 módulos:

```text
36 × 2048 × 4 = 294.912 bytes
```

Es decir, otra vez parece que el reporte excluye `c_fc`, pero el código lo incluye.

---

# Conclusión sobre EXP-11/EXP-12

El benchmark actual tiene al menos cuatro problemas serios:

1. **LoRA probablemente entrena parámetros extra de GPT-2.**
2. **El tamaño reportado de LoRA no coincide con los targets usados.**
3. **El tamaño reportado de SpecRAMA tampoco coincide con los targets usados.**
4. **La comparación “5.2× menos parámetros” no está sustentada por el código copiado.**

Esto no mata la idea de SpecRAMA. De hecho, puede que SpecRAMA sea más interesante de lo que parece si se compara contra un LoRA realmente congelado. Pero los números actuales no son publicables tal cual.

---

# Qué corregiría inmediatamente

## 1. Congelar todo el modelo antes de entrenar LoRA

Modificar `inject_lora_in_model()` así:

```python
def inject_lora_in_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: float = 16.0,
) -> List[LoRALinear]:

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    injected = []

    def _inject(module: nn.Module):
        for name, child in list(module.named_children()):
            is_target = isinstance(child, nn.Linear) or child.__class__.__name__ in ["Conv1D", "Linear"]
            if is_target and any(target in name for target in target_modules):
                wrapped = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, name, wrapped)
                injected.append(wrapped)
            else:
                _inject(child)

    _inject(model)

    # Unfreeze only LoRA matrices
    for m in injected:
        m.lora_A.requires_grad = True
        m.lora_B.requires_grad = True

    return injected
```

O, si se quiere mantener la función flexible, crear una utilidad separada:

```python
def mark_only_lora_as_trainable(model):
    for p in model.parameters():
        p.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
```

Y llamarla en EXP-11/EXP-12.

---

## 2. Imprimir conteos reales en cada brazo

Antes de entrenar cada modelo:

```python
def print_trainable_report(model, title=""):
    total = 0
    print(f"\n=== Trainable parameter report: {title} ===")
    for name, p in model.named_parameters():
        if p.requires_grad:
            n = p.numel()
            total += n
            print(f"{name:80s} {n:12d}")
    print(f"TOTAL trainable params: {total:,}")
    print(f"TOTAL FP32 bytes: {total * 4:,}")
    print(f"TOTAL FP32 MB: {total * 4 / 1e6:.3f} MB")
```

Esto habría detectado inmediatamente si `wte`/`wpe` estaban entrenándose.

---

## 3. Decidir los targets y hacerlos consistentes

Si quieres reportar 294.9 KB SpecRAMA y 1.55 MB LoRA, entonces usa:

```python
targets = ["c_attn", "c_proj"]
```

Eso incluye:

```text
attn.c_attn
attn.c_proj
mlp.c_proj
```

pero excluye:

```text
mlp.c_fc
```

Si quieres incluir también `c_fc`, entonces los tamaños deben actualizarse:

```text
SpecRAMA 32×32, multiplicativo+aditivo: ~393 KB
LoRA r=4: ~2.36 MB
```

La ratio seguiría siendo parecida:

```text
2.36 MB / 0.393 MB ≈ 6.0×
```

pero los números deben ser correctos.

---

## 4. Rehacer EXP-12 con LoRA realmente congelado

El experimento mínimo nuevo debería ser:

```text
NF4 fake-quantized GPT-2 Small
WikiText-2 train/validation/test
500 steps
seed = 42, 43, 44
```

Comparar:

```text
LoRA r=1
LoRA r=2
LoRA r=4
SpecRAMA 16×16
SpecRAMA 32×32
```

Todos con:

```text
solo adaptadores entrenables
mismos target modules
mismo número de pasos
mismo train split
LR elegido en validation, no test
```

Y reportar:

```text
trainable params reales
adapter bytes reales
PPL validation
PPL test
bpt test
tokens/sec
peak VRAM
```

---

# Evaluación actualizada del candidato

Mi evaluación de la persona sigue siendo positiva, pero la evaluación de los resultados publicados baja.

## Señales fuertes

- Implementa ideas no triviales.
- Entiende PEFT, cuantización y arquitectura GPT-2 suficientemente bien para prototipar.
- Tiene intuición original.
- Corrigió honestamente un baseline LoRA mal tuneado.
- El código es legible y auditable.

## Señales preocupantes

- El benchmark tiene errores de protocolo.
- Los tamaños reportados no coinciden con el código.
- El baseline LoRA probablemente no está congelado.
- Los resultados se presentan con más fuerza de la que permite el arnés.
- Hay riesgo de “research storytelling” por delante de la validación.

---

# Veredicto de reclutamiento

Mi decisión como reclutador sería:

```text
Entrevista técnica: sí.
Oferta directa: no.
Reproducción controlada obligatoria: sí.
```

Lo pondría en la categoría:

> **High-upside research engineer con necesidad de disciplina experimental.**

No veo esto como un perfil débil. Al contrario: veo un perfil con mucha capacidad inventiva. Pero para un laboratorio frontier, el siguiente paso sería comprobar si puede transformar una idea prometedora en un resultado limpio, reproducible y defendible ante reviewers duros.

---

# Archivo que quiero leer ahora

Ahora quiero ver:

```text
spec_rama/utils.py
```

En particular:

```python
inject_spec_rama_in_model
count_trainable_parameters
merge_spec_rama_modules
```

Motivo: necesitamos comprobar si la utilidad oficial de SpecRAMA sí congela correctamente, cómo calcula el tamaño del adaptador y si coincide o no con los números reportados.


---


import torch
import torch.nn as nn
from typing import List, Tuple
from .layers import SpecRAMALinear
from .shared_layers import SharedSpecRAMALinear

def inject_spec_rama_in_model(
    model: nn.Module,
    target_modules: List[str],
    transform_type: str = "dct",
    core_size: Tuple[int, int] = (8, 8),
    alpha_m: float = 1.0,
    alpha_a: float = 8.0,
    permutation_method: str = "tsp",
    use_multiplicative: bool = True,
    use_additive: bool = True,
) -> List[SpecRAMALinear]:
    """
    Recursively finds target Linear layers in a PyTorch model and wraps them with SpecRAMALinear.
    Returns a list of injected layers.
    """
    injected_layers = []

    def _inject(module: nn.Module):
        for name, child in list(module.named_children()):
            is_target_layer = isinstance(child, nn.Linear) or child.__class__.__name__ in ["Conv1D", "Linear"]
            if is_target_layer and any(target in name for target in target_modules):
                wrapped = SpecRAMALinear(
                    base_layer=child,
                    transform_type=transform_type,
                    core_size=core_size,
                    alpha_m=alpha_m,
                    alpha_a=alpha_a,
                    permutation_method=permutation_method,
                    use_multiplicative=use_multiplicative,
                    use_additive=use_additive,
                )
                setattr(module, name, wrapped)
                injected_layers.append(wrapped)
            else:
                _inject(child)

    _inject(model)
    return injected_layers


def merge_spec_rama_modules(model: nn.Module):
    """
    Recursively calls merge() on all SpecRAMALinear and SharedSpecRAMALinear modules in a model.
    """
    for module in model.modules():
        if isinstance(module, (SpecRAMALinear, SharedSpecRAMALinear)):
            module.merge()


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Returns (trainable_params, total_params, percentage_trainable).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = (trainable / total * 100.0) if total > 0 else 0.0
    return trainable, total, ratio


---

## Auditoría de `spec_rama/utils.py`

Este archivo confirma una parte importante de la evaluación anterior:

> La utilidad oficial `inject_spec_rama_in_model()` tampoco congela todo el modelo.  
> Por tanto, si otros experimentos usan esta utilidad sin congelado manual posterior, también pueden estar entrenando parámetros extra del modelo base.

Sin embargo, en **EXP-11**, los brazos SpecRAMA que me copiaste sí hacen congelado manual después de la inyección:

```python
for p in model_nf4_spec.parameters():
    p.requires_grad = False

for m in inject_nf4_spec:
    if m.core_m is not None:
        m.core_m.requires_grad = True
    if m.core_a is not None:
        m.core_a.requires_grad = True
```

Así que la contaminación más grave sigue estando en **LoRA**, no necesariamente en SpecRAMA dentro de EXP-11.

---

# Hallazgos principales

## 1. `inject_spec_rama_in_model()` es insegura como API experimental

La función:

```python
def inject_spec_rama_in_model(...):
    injected_layers = []

    def _inject(module: nn.Module):
        ...
            if is_target_layer and any(target in name for target in target_modules):
                wrapped = SpecRAMALinear(...)
                setattr(module, name, wrapped)
                injected_layers.append(wrapped)
```

envuelve módulos objetivo, pero no hace:

```python
for p in model.parameters():
    p.requires_grad = False
```

ni luego reactiva solo:

```python
core_m
core_a
```

Esto significa que un usuario que use la API de forma natural podría entrenar:

- embeddings;
- positional embeddings;
- layer norms;
- `lm_head`;
- otros parámetros no envueltos.

La API debería ofrecer una opción clara:

```python
freeze_base: bool = True
```

y por defecto debería congelar todo para PEFT.

---

## 2. EXP-11 no usa esta utilidad para SpecRAMA, pero sí debería usar una versión equivalente para LoRA

En EXP-11, SpecRAMA se inyecta manualmente y luego se congela correctamente.

En cambio, LoRA se inyecta con:

```python
inject_lora_in_model(...)
```

y esa función, como vimos, solo congela las capas envueltas, no el resto de GPT-2.

Así que mi diagnóstico queda reforzado:

> El brazo SpecRAMA de EXP-11 parece limpio respecto a parámetros entrenables.  
> El brazo LoRA probablemente no lo está.

Eso implica que el baseline LoRA reportado como:

```text
1.55 MB
```

probablemente no corresponde al número real de parámetros entrenados.

---

## 3. `count_trainable_parameters()` existe, pero no se está usando donde importa

La función:

```python
def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = ...
    return trainable, total, ratio
```

es útil, pero en EXP-11/EXP-12 no vi que se llame antes de entrenar cada brazo.

Esto habría detectado inmediatamente si LoRA estaba entrenando `wte`, `wpe`, layer norms o `lm_head`.

Para un benchmark serio, pondría asserts obligatorios:

```python
trainable, total, ratio = count_trainable_parameters(model_nf4_lora)
print("LoRA trainable:", trainable)

assert trainable == expected_lora_params
```

o al menos:

```python
assert trainable < 1_000_000
```

para evitar accidentalmente entrenar decenas de millones de parámetros.

---

## 4. `count_trainable_parameters()` no cuenta buffers, y eso importa mucho en SpecRAMA

SpecRAMA registra permutaciones como buffers:

```python
row_perm
col_perm
row_inv_perm
col_inv_perm
```

Estos no son parámetros entrenables, pero sí forman parte del `state_dict` por defecto porque `register_buffer(..., persistent=True)` es el comportamiento estándar.

Por tanto, el tamaño real de un adaptador SpecRAMA no es solo:

```text
core_m + core_a
```

También hay que decidir si se cuentan:

```text
permutaciones
bases DCT/Walsh, si se usan
metadata
```

En Wavelet no hay matrices DCT/Walsh, pero sí permutaciones.

Esto es importante porque las permutaciones pueden ser más grandes que los cores.

Ejemplo aproximado para GPT-2 Small con 48 módulos objetivo:

```text
core 32×32 multiplicativo+aditivo:
48 × 2 × 32 × 32 × 4 bytes ≈ 393 KB
```

Pero las permutaciones, si se guardan como `int64` y se guardan cuatro por módulo:

```text
row_perm + col_perm + row_inv_perm + col_inv_perm
```

pueden sumar del orden de **megabytes**, no kilobytes.

Si el claim es:

```text
294.9 KB adapter
```

debe aclarar:

```text
294.9 KB = solo parámetros entrenables del core
```

No necesariamente:

```text
294.9 KB = checkpoint real del adaptador
```

Para deployment real, esto importa.

---

# Cambio recomendado en la API

Yo reescribiría `inject_spec_rama_in_model()` así:

```python
def inject_spec_rama_in_model(
    model: nn.Module,
    target_modules: List[str],
    transform_type: str = "dct",
    core_size: Tuple[int, int] = (8, 8),
    alpha_m: float = 1.0,
    alpha_a: float = 8.0,
    permutation_method: str = "tsp",
    use_multiplicative: bool = True,
    use_additive: bool = True,
    freeze_base: bool = True,
) -> List[SpecRAMALinear]:

    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    injected_layers = []

    def _inject(module: nn.Module):
        for name, child in list(module.named_children()):
            is_target_layer = isinstance(child, nn.Linear) or child.__class__.__name__ in ["Conv1D", "Linear"]

            if is_target_layer and any(target in name for target in target_modules):
                wrapped = SpecRAMALinear(
                    base_layer=child,
                    transform_type=transform_type,
                    core_size=core_size,
                    alpha_m=alpha_m,
                    alpha_a=alpha_a,
                    permutation_method=permutation_method,
                    use_multiplicative=use_multiplicative,
                    use_additive=use_additive,
                )
                setattr(module, name, wrapped)
                injected_layers.append(wrapped)
            else:
                _inject(child)

    _inject(model)

    if freeze_base:
        for layer in injected_layers:
            if layer.core_m is not None:
                layer.core_m.requires_grad = True
            if layer.core_a is not None:
                layer.core_a.requires_grad = True

    return injected_layers
```

Y haría lo mismo para LoRA.

---

# Veredicto actualizado

El patrón ya está claro:

## El candidato

Sigue pareciendo un candidato interesante:

- creativo;
- capaz de implementar ideas propias;
- buen comunicador;
- con intuición de compresión/PEFT/cuántización;
- suficientemente honesto para corregir un baseline mal tuneado.

## El benchmark

Pero el benchmark SpecRAMA, tal como está, **no está aún en estado publicable**.

Los problemas principales son:

1. bug de reproducción en `permutation.py`;
2. test set parcial reportado como 280k tokens;
3. fake quantization presentada como 4-bit real;
4. tamaños de adaptador inconsistentes con los targets;
5. LoRA probablemente entrena parámetros extra;
6. ausencia de múltiples semillas;
7. ausencia de validación separada;
8. conteo de parámetros no auditado;
9. buffers/permutaciones no incluidos en tamaño real del adaptador.

Mi decisión como reclutador no cambia mucho:

```text
Entrevista: sí.
Oferta directa: no.
Prueba de reproducción: obligatoria.
```

Lo evaluaría como:

> **High-upside research engineer, con fuerte creatividad, pero necesita disciplina experimental de nivel frontier.**

---

# Archivo que quiero leer ahora

Ahora quiero ver uno de estos dos, en orden de prioridad:

```text
benchmarks/exp04_equal_parameter_head2head.py
```

Motivo: necesito ver cómo se hizo la comparación “iso-parámetro” previa. Ese experimento probablemente revela si los tamaños/targets se contaban correctamente antes de EXP-11/12.

Si prefieres copiar otro archivo más corto, entonces copia:

```text
benchmarks/exp03_spectral_core_scaling_sweep.py
```

Pero mi primera elección es:

```text
benchmarks/exp04_equal_parameter_head2head.py
```


---


# Auditoría de `quantization.py`, EXP-03 y EXP-04

La evidencia nueva permite separar con bastante claridad tres líneas que hasta ahora aparecían mezcladas:

1. **Spec-RAMA espectral en FP32** — EXP-03 y EXP-04.
2. **Emulación de pesos NF4 reconstruidos en float** — EXP-11/EXP-12.
3. **Cuantizador espectral jerárquico offline** — `HierarchicalSpectralQuantizer`.

La tercera línea todavía no constituye una implementación real de inferencia cuantizada y, además, contiene errores que impiden usarla tal como está. En cambio, EXP-03 y EXP-04 sí tienen una estructura experimental más coherente, aunque sus afirmaciones de “presupuesto exactamente igual” deben matizarse.

## Veredicto resumido

> El candidato tiene una línea de investigación consistente y varios prototipos funcionales, pero el repositorio presenta una separación insuficiente entre resultados históricos, código actual y formatos de memoria reales.

La valoración sigue siendo suficientemente positiva para recomendar entrevista técnica, pero no aceptaría todavía las cifras de compresión como resultados de sistemas sin una implementación de almacenamiento y ejecución cuantizada verificable.

---

# 1. `HierarchicalSpectralQuantizer` no es todavía un cuantizador runtime

La clase transforma una matriz:

\[
W \rightarrow W_{\pi} \rightarrow C
\]

donde:

- \(W_{\pi}\) es la matriz permutada;
- \(C\) es la representación DCT o Walsh;
- una región de baja frecuencia se cuantiza a más bits;
- el resto se cuantiza a menos bits.

La idea es razonable. Sin embargo, el código es un **cuantizador offline que devuelve un payload**, no una capa cuantizada integrada en un modelo.

No implementa:

- almacenamiento bit-packed;
- `nn.Module` cuantizado;
- multiplicación directamente sobre coeficientes cuantizados;
- kernels CUDA/Triton;
- integración con `Linear4bit`;
- ejecución sin reconstruir la matriz completa;
- serialización de escalas y permutaciones con un formato definido.

El método de dequantización termina devolviendo:

```python
weight_reconstructed
```

como una matriz densa `float32`.

Por tanto, no demuestra ahorro de VRAM ni reducción de coste de inferencia.

La descripción correcta sería:

> Prototipo de cuantización espectral jerárquica offline con reconstrucción densa en float32.

No debería presentarse todavía como un método de inferencia cuantizada.

---

# 2. `avg_bits` es una cifra teórica, no el tamaño físico del payload

El código calcula:

```python
avg_bits = (
    core_weights * self.core_bits
    + rest_weights * self.rest_bits
) / total_weights
```

Con los valores por defecto:

```python
core_bits = 8
rest_bits = 4
core_ratio = 0.0625
```

el promedio teórico sería aproximadamente:

\[
0.0625 \times 8 + 0.9375 \times 4 = 4.25
\]

bits por coeficiente.

Pero el almacenamiento real usa:

```python
core_q = ... .to(torch.uint8)
rest_q = ... .to(torch.uint8)
```

Cada elemento ocupa un byte completo. Los valores de 4 bits no se empaquetan en nibbles.

Así, el payload almacena aproximadamente:

```text
8 bits por coeficiente para el core
8 bits por coeficiente para el resto
```

no 4,25 bits por coeficiente.

Además, hay sobrecostes por:

```python
row_perm
col_perm
```

que son tensores `torch.long`, normalmente de 64 bits por índice, además de:

- `core_min`;
- `core_scale`;
- `rest_min`;
- `rest_scale`;
- forma;
- configuración de transformación.

Por consiguiente, `avg_bits` no es una medición de tamaño de checkpoint. Es sólo el promedio lógico de bits que el algoritmo pretende utilizar si se implementa empaquetado real.

## El caso `rest_bits=0` tampoco está implementado

La documentación menciona:

> 4-bit / 2-bit / 0-bit cutoff

Pero el código siempre intenta cuantizar el resto:

```python
rest_scale = (
    rest_max - rest_min
) / (2**self.rest_bits - 1) + 1e-8
```

Con `rest_bits=0`, el denominador es cero. Además, no se elimina ni se ignora el resto de coeficientes: se sigue almacenando un `rest_q`.

Para soportar un cutoff real de cero bits habría que hacer algo como:

```python
if self.rest_bits == 0:
    rest_q = None
    rest_min = 0.0
    rest_scale = 0.0
```

y reconstruir los coeficientes restantes como cero.

También faltan validaciones para:

- `core_bits > 8`;
- `rest_bits > 8`;
- valores negativos;
- regiones vacías;
- matrices demasiado pequeñas para separar core y resto.

---

# 3. El cuantizador falla con su configuración por defecto

El constructor utiliza:

```python
permutation_method="bipartite_tsp"
```

y llama a:

```python
compute_2d_permutations(weight, method=self.permutation_method)
```

Pero ya observamos que en `permutation.py` aparece:

```python
return compute_joint_bipartite_2d_permutation(weight, num_iters=2)
```

mientras la función acepta:

```python
max_iters
```

Esto produce:

```text
TypeError: unexpected keyword argument 'num_iters'
```

Por tanto:

```python
HierarchicalSpectralQuantizer()
```

no es utilizable con su configuración por defecto en el snapshot entregado.

Hay una distinción importante:

- EXP-03 y EXP-04 usan el default `"tsp"` de `inject_spec_rama_in_model`, por lo que no activan este error.
- EXP-11 usa explícitamente `"bipartite_tsp"` y sí queda afectado.
- `HierarchicalSpectralQuantizer` también queda afectado por defecto.

La corrección inmediata es:

```python
return compute_joint_bipartite_2d_permutation(
    weight,
    max_iters=2
)
```

Pero, para un resultado de investigación reproducible, habría que identificar el commit histórico con el que se ejecutó cada experimento.

---

# 4. Problema de dispositivo y dtype

Las bases espectrales se crean así:

```python
B_out = get_dct_matrix_1d(out_dim)
B_in = get_dct_matrix_1d(in_dim)
```

Estas matrices se crean en CPU y `float32`.

Si `weight` está en CUDA, esta operación:

```python
coeff = torch.matmul(
    B_out,
    torch.matmul(w_perm, B_in.t())
)
```

puede fallar por mezclar CPU y CUDA.

También puede fallar si `weight` está en `float16` o `bfloat16`, porque las bases siguen siendo `float32`.

La implementación debería hacer explícitamente:

```python
B_out = get_dct_matrix_1d(out_dim).to(
    device=weight.device,
    dtype=weight.dtype
)
B_in = get_dct_matrix_1d(in_dim).to(
    device=weight.device,
    dtype=weight.dtype
)
```

o calcular la transformación en `float32` y restaurar después el dtype deseado.

Este problema refuerza que el cuantizador no ha sido integrado todavía en el mismo flujo GPU de los benchmarks.

---

# 5. La variante Walsh no es ortogonal para las dimensiones de GPT-2

`get_walsh_matrix_1d` exige dimensiones potencia de dos. Para otras dimensiones hace:

```python
p = 2 ** math.ceil(math.log2(N))
H_full = get_walsh_matrix_1d(p)
return H_full[:N, :N]
```

Esto no produce una matriz Walsh-Hadamard ortogonal de tamaño \(N\times N\). Es simplemente un bloque recortado de una matriz mayor.

Para GPT-2 aparecen dimensiones como:

- 768;
- 2304;
- 3072.

Ninguna es potencia de dos. Por ejemplo:

- 768 se recorta desde 1024;
- 2304 se recorta desde 4096;
- 3072 se recorta desde 4096.

En general:

\[
H_N^T H_N \neq I
\]

para esas matrices truncadas.

Por tanto, esta inversión:

```python
w_perm = B_out.t() @ coeff @ B_in
```

no es una inversión exacta. Incluso con cuantización perfecta, la ruta:

```text
quantize_matrix -> dequantize_matrix
```

puede modificar significativamente los pesos en la variante Walsh.

La implementación correcta tendría que:

- rellenar las dimensiones hasta potencias de dos;
- transformar en la dimensión ampliada;
- conservar sólo la región válida;
- definir una inversión consistente;

o utilizar otra base ortogonal para dimensiones arbitrarias.

La variante DCT no tiene este problema porque la matriz DCT cuadrada sí es ortogonal con la normalización utilizada.

---

# 6. No hay evidencia de que este cuantizador genere los resultados NF4

El código de `HierarchicalSpectralQuantizer` no se utiliza en los fragmentos de EXP-03 ni EXP-04.

Esos experimentos hacen:

```python
GPT2LMHeadModel.from_pretrained("gpt2")
```

y aplican Spec-RAMA directamente sobre pesos FP32.

Por tanto:

- EXP-03 es un benchmark espectral FP32;
- EXP-04 es una comparación de presupuestos en FP32;
- ninguno valida la cuantización jerárquica.

Los resultados NF4 de EXP-11/EXP-12 usan otra función:

```python
quantize_blockwise_nf(...)
```

que reconstruye el peso cuantizado inmediatamente como float.

Estas son líneas separadas y deberían aparecer separadas en la documentación:

| Línea | Implementación actual |
|---|---|
| Spec-RAMA FP32 | Sí, prototipo funcional |
| Emulación NF4 | Sí, pesos reconstruidos en float |
| NF4 real empaquetado | No |
| Cuantización espectral jerárquica | Payload offline, sin runtime |
| Inferencia espectral cuantizada | No demostrada |

---

# 7. EXP-03 y EXP-04 sí congelan correctamente el modelo

Este punto corrige parcialmente la preocupación anterior sobre LoRA.

En ambos experimentos aparece:

```python
for p in model.parameters():
    p.requires_grad = False
```

antes de inyectar los adaptadores.

Por tanto, en EXP-03 y EXP-04:

- LoRA sólo entrena `lora_A` y `lora_B`;
- Spec-RAMA sólo entrena sus núcleos;
- embeddings y LayerNorm permanecen congelados.

Esto no ocurre en EXP-11/EXP-12, donde el modelo LoRA se inyecta sin congelación global previa.

La conclusión debe dividirse:

| Experimento | Estado del baseline LoRA |
|---|---|
| EXP-03 | Congelación global correcta |
| EXP-04 | Congelación global correcta |
| EXP-11 | No congelación global en el código mostrado |
| EXP-12 | No congelación global en el código mostrado |

Los resultados de EXP-03 y EXP-04 son, por tanto, más sólidos como comparaciones PEFT internas que el baseline LoRA de EXP-11/EXP-12.

---

# 8. EXP-04 es casi iso-paramétrico, pero no exactamente

EXP-04 utiliza:

```python
targets = ["c_attn", "c_proj"]
```

Con el selector actual, esto incluye:

```text
attn.c_attn
attn.c_proj
mlp.c_proj
```

Es decir, tres módulos por bloque. La selección es consistente entre LoRA y Spec-RAMA, aunque probablemente accidental para `mlp.c_proj`.

## Tier de rango 1

LoRA rank 1:

\[
101.376 \text{ parámetros}
\]

Spec-RAMA con núcleo \(37\times38\):

\[
36 \times 2 \times 37 \times 38
=101.232
\]

Diferencia:

\[
144 \text{ parámetros}
\]

## Tier de rango 4

LoRA rank 4:

\[
405.504 \text{ parámetros}
\]

Spec-RAMA con núcleo \(75\times75\):

\[
36 \times 2 \times 75 \times 75
=405.000
\]

Diferencia:

\[
504 \text{ parámetros}
\]

La diferencia es pequeña —aproximadamente 0,1%—, pero el título:

> EXACT ISO-PARAMETER

es demasiado fuerte. Sería más preciso:

> comparación con presupuesto de parámetros prácticamente igualado.

Además, el tamaño mostrado depende de las unidades. Para 405.000 parámetros de Spec-RAMA:

```text
405.000 × 4 = 1.620.000 bytes
```

El script utiliza unidades binarias:

```python
size_bytes / (1024 * 1024)
```

mientras otros documentos presentan tamaños decimales. Esto explica discrepancias como:

- 294.912 bytes;
- 288,0 KiB;
- 294,9 KB decimales.

El repositorio debería elegir una convención y utilizarla de forma consistente.

---

# 9. La selección de módulos sigue siendo ambigua

En EXP-03/04:

```python
targets = ["c_attn", "c_proj"]
```

no significa “proyecciones de atención”. Significa:

```text
todos los hijos cuyo nombre local contiene c_attn o c_proj
```

Por tanto, incluye `mlp.c_proj`.

Si esa era la intención, debe escribirse explícitamente. Si no lo era, el benchmark está adaptando una capa MLP no declarada.

Una versión reproducible debería usar rutas completas:

```python
targets = [
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_proj",
]
```

o, si se desean también las capas fully connected:

```python
targets = [
    "attn.c_attn",
    "attn.c_proj",
    "mlp.c_fc",
    "mlp.c_proj",
]
```

Y el script debe imprimir:

```python
for name, module in model.named_modules():
    ...
```

para demostrar exactamente qué se envolvió.

---

# 10. El benchmark sigue usando sólo una fracción del test set

EXP-03 y EXP-04 llaman a:

```python
prepare_wikitext_data(
    block_size=256,
    max_train_samples=400,
    max_test_samples=100
)
```

La evaluación se hace sobre, como máximo:

```text
100 × 256 = 25.600 tokens de entrada
```

No sobre el test set completo de WikiText-2.

Esto no invalida necesariamente una comparación interna si todos los brazos ven exactamente los mismos datos, pero sí impide presentar el número como una evaluación completa del corpus.

Además:

- no se utiliza validation set;
- los hiperparámetros se fijan manualmente;
- sólo se muestra una ejecución;
- no hay desviaciones estándar;
- no se controlan explícitamente las semillas antes de inicializar los adaptadores.

El resultado debe describirse como un benchmark exploratorio sobre un subconjunto fijo.

---

# Evaluación de la madurez del proyecto

## Lo que mejora la valoración

EXP-03 y EXP-04 muestran varias decisiones correctas:

- congelación global antes de inyectar adaptadores;
- comparación de LoRA y Spec-RAMA en presupuestos cercanos;
- comprobación de PPL antes y después de fusionar;
- uso de `count_trainable_parameters`;
- una progresión de tamaños de núcleo;
- separación entre wavelet y DCT.

Estas pruebas indican que el candidato no está únicamente acumulando ideas: también intenta diseñar comparaciones controladas.

## Lo que sigue siendo insuficiente

La implementación de cuantización no sustenta todavía las afirmaciones de ahorro físico:

- los bits no están empaquetados;
- se reconstruye una matriz densa;
- las permutaciones ocupan memoria no contabilizada;
- no hay runtime cuantizado;
- el método Walsh no es ortogonal para GPT-2;
- el default de permutación falla;
- no hay benchmark de memoria real ni velocidad frente a una implementación NF4 estándar.

# Veredicto editorial actualizado

La afirmación más sólida que puede publicarse ahora es:

> EXP-03 y EXP-04 proporcionan evidencia exploratoria de que una parametrización espectral puede aproximar a LoRA bajo presupuestos de parámetros casi iguales en GPT-2 Small y WikiText-2. La evidencia se limita a modelos FP32, un subconjunto del dataset y una sola ejecución. La implementación de cuantización jerárquica todavía es offline y no demuestra almacenamiento o ejecución a 4 bits.

No publicaría aún:

- “cuantización espectral real”;
- “4,25 bits por peso”;
- “ahorro de VRAM”;
- “inferencia cuantizada eficiente”;
- “presupuesto exactamente igual”;
- “Walsh ortogonal para modelos GPT-2”.

## Evaluación del candidato

| Dimensión | Valoración |
|---|---:|
| Originalidad | 8,5/10 |
| Capacidad de prototipado | 8/10 |
| Diseño de comparaciones | 6,5/10 |
| Rigor de cuantificación de memoria | 3,5/10 |
| Reproducibilidad | 4/10 |
| Ingeniería de kernels/runtime | 4/10 |
| Potencial de investigación | 8/10 |

Mi recomendación sigue siendo:

> **Avanzar a entrevista técnica, pero centrarla en reproducibilidad, formatos de memoria y corrección de benchmarks.**

La prueba práctica que propondría al candidato sería:

1. corregir el selector de módulos;
2. corregir `num_iters`;
3. congelar todos los modelos de forma uniforme;
4. implementar bit-packing real para 4 bits;
5. demostrar `quantize → dequantize` con error medido;
6. comparar memoria física y velocidad;
7. repetir EXP-04 con tres semillas y validation/test separados.

El siguiente archivo que quiero leer es:

```text
spec-rama/shared_layers.py
```

Es necesario para auditar la afirmación de adaptadores sub-KB y comprobar si el parámetro compartido se cuenta correctamente en EXP-04.



---


## Actualización de evaluación tras leer `quantization.py`, `exp04_equal_parameter_head2head.py` y `exp03_spectral_core_scaling_sweep.py`

Esta copia aclara bastante la historia. Mi evaluación queda más matizada:

> **EXP-03/EXP-04 son bastante más limpios que EXP-11/EXP-12 en cuanto a congelado y conteo de parámetros.**  
> Pero también muestran que varios claims recientes mezclan protocolos distintos: targets distintos, tamaños heredados de EXP-04, y código de EXP-11/12 que no coincide con esos tamaños.

La idea SpecRAMA sigue pareciendo prometedora. El problema principal no es la idea, sino la **consistencia experimental**.

---

# 1. Hallazgo positivo: EXP-04 sí congela bien el modelo

En `run_exp04()` haces:

```python
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
for p in model.parameters():
    p.requires_grad = False
```

antes de inyectar LoRA o SpecRAMA.

Eso significa que, en EXP-04:

```text
LoRA: solo matrices LoRA entrenables.
SpecRAMA: solo cores entrenables.
```

Este experimento es mucho más limpio que EXP-11/12, donde el brazo LoRA no congelaba explícitamente el resto de GPT-2.

Esto rehabilita parcialmente la comparación iso-parámetro antigua.

---

# 2. EXP-04 explica de dónde salen los tamaños “294.9 KB” y “1.55 MB”

En EXP-04 los targets son:

```python
targets = ["c_attn", "c_proj"]
```

Eso incluye:

```text
attn.c_attn
attn.c_proj
mlp.c_proj
```

pero excluye:

```text
mlp.c_fc
```

En GPT-2 Small son:

```text
3 módulos por bloque × 12 bloques = 36 módulos
```

Con SpecRAMA `32×32`, multiplicativo + aditivo:

```text
32 × 32 × 2 cores × 36 módulos = 73.728 parámetros
73.728 × 4 bytes = 294.912 bytes
```

Eso coincide exactamente con:

```text
294.9 KB
```

Y LoRA `r=4` con esos targets:

```text
405.504 parámetros ≈ 1.55 MiB
```

Así que el número **294.9 KB vs 1.55 MB** viene de un protocolo con:

```python
targets = ["c_attn", "c_proj"]
```

no de:

```python
targets = ["c_attn", "c_proj", "c_fc"]
```

como en EXP-11/12.

---

# 3. Problema: EXP-11/12 mezcla targets distintos

En EXP-11/12 usas:

```python
targets = ["c_attn", "c_proj", "c_fc"]
```

Eso incluye también:

```text
mlp.c_fc
```

Entonces los tamaños deberían cambiar.

Para SpecRAMA `32×32`:

```text
4 módulos por bloque × 12 bloques = 48 módulos

32 × 32 × 2 × 48 = 98.304 parámetros
98.304 × 4 = 393.216 bytes
```

Es decir:

```text
~393 KB, no 294.9 KB
```

Para LoRA `r=4`:

```text
~589.824 parámetros
~2.36 MB decimal / ~2.25 MiB
```

no:

```text
405K / 1.55 MB
```

Por tanto, hay que elegir:

### Opción A — mantener el claim 294.9 KB vs 1.55 MB

Entonces EXP-11/12 debe usar:

```python
targets = ["c_attn", "c_proj"]
```

### Opción B — mantener EXP-11/12 con `c_fc`

Entonces la tabla debe actualizar tamaños a:

```text
SpecRAMA: ~393 KB
LoRA r=4: ~2.36 MB
```

La ratio seguiría siendo favorable, incluso alrededor de `6×`, pero los números deben ser consistentes.

---

# 4. EXP-04 sí hace una comparación iso-parámetro bastante elegante

Esta parte es buena.

Para LoRA `r=1`:

```text
~101.376 parámetros
```

Para SpecRAMA `37×38`:

```text
37 × 38 × 2 × 36 = 101.232 parámetros
```

Muy cerca.

Para LoRA `r=4`:

```text
405.504 parámetros
```

Para SpecRAMA `75×75`:

```text
75 × 75 × 2 × 36 = 405.000 parámetros
```

También muy cerca.

Esto es una buena señal: el candidato sabe construir una comparación iso-parámetro real cuando se concentra en ello.

---

# 5. Pero EXP-03/04 siguen siendo benchmarks exploratorios, no definitivos

Aunque EXP-03/04 están mejor que EXP-11/12 en congelado, siguen teniendo limitaciones:

## A. Evalúan solo 100 bloques de test

```python
max_test_samples=100
block_size=256
```

Eso son aproximadamente:

```text
25.600 tokens
```

No el test completo de WikiText-2.

Por tanto, cualquier claim que diga:

```text
WikiText-2 Test Set, 280.000 tokens
```

no aplica a este código.

## B. No hay validation split

Se entrena y se reporta en test. Para un paper o benchmark serio hace falta:

```text
train → entrenamiento
validation → selección de LR/core/alpha/targets
test → evaluación final congelada
```

## C. Una sola semilla

No hay `seed` explícito en `train_on_dataset()` de EXP-03/04. En LoRA, la matriz `A` se inicializa aleatoriamente:

```python
nn.init.kaiming_uniform_(self.lora_A, ...)
```

Así que el resultado puede variar. Haría falta:

```text
3-5 semillas
media ± std
```

## D. Learning rates no igualmente optimizados

En EXP-04:

```python
LoRA: lr=1e-3
SpecRAMA: lr=1e-2
```

Esto puede ser razonable si SpecRAMA tolera más LR, pero para una comparación definitiva habría que hacer sweep por método usando validation.

---

# 6. Importante: el tamaño de adaptador sigue contando solo parámetros entrenables

En EXP-03/04 haces:

```python
size_bytes = trainable * 4
```

Eso cuenta solo parámetros entrenables.

Pero SpecRAMA registra buffers:

```text
row_perm
col_perm
row_inv_perm
col_inv_perm
```

Y en DCT/Walsh también registra bases:

```text
B_out
B_in
```

Por tanto:

```text
adapter size = trainable params × 4
```

no es necesariamente igual a:

```text
state_dict real del adaptador
```

Para Wavelet, el problema principal son las permutaciones. Para DCT/Walsh, el problema puede ser mucho mayor porque las matrices DCT/Walsh completas pueden ser enormes si se guardan como buffers persistentes.

La solución limpia sería reportar tres tamaños:

```text
1. trainable core size
2. deployable adapter size, incluyendo permutaciones comprimidas
3. full state_dict size
```

Y, si las permutaciones se pueden recomputar desde el modelo base, decirlo explícitamente.

---

# 7. Auditoría rápida de `HierarchicalSpectralQuantizer`

El cuantizador jerárquico es una línea interesante, pero actualmente parece más un prototipo que una implementación lista para claims de memoria.

Problemas principales:

## A. `rest_bits=4` no está realmente empaquetado

Aunque se calcula:

```python
avg_bits = ...
```

el código guarda:

```python
rest_q = ... .to(torch.uint8)
```

Eso usa 8 bits por coeficiente en memoria real, no 4.

Para poder decir “4-bit storage” hace falta bit-packing real.

## B. Las permutaciones no están incluidas en el coste

El payload guarda:

```python
row_perm
col_perm
```

Como tensores `int64`, pueden tener coste no trivial.

## C. Posible bug con `bipartite_tsp`

El default es:

```python
permutation_method="bipartite_tsp"
```

pero en el archivo anterior `compute_2d_permutations()` llamaba:

```python
compute_joint_bipartite_2d_permutation(weight, num_iters=2)
```

cuando la función aceptaba:

```python
max_iters
```

Si esa versión es la actual, este cuantizador falla por `TypeError`.

## D. DCT/Walsh full matrices no escalan

Para matrices grandes, construir:

```python
B_out = get_dct_matrix_1d(out_dim)
B_in = get_dct_matrix_1d(in_dim)
```

puede ser muy costoso. Para modelos grandes necesitarías transformadas implícitas o kernels tipo FFT/DCT/FWHT, no matrices densas.

## E. Walsh para tamaños no potencia de 2 no es estrictamente ortogonal

La función `get_walsh_matrix_1d(N)` recorta una Hadamard mayor si `N` no es potencia de 2. Ese recorte no garantiza una base ortonormal completa. Esto debilita claims tipo Parseval para Walsh en dimensiones como 768.

---

# Evaluación del candidato, actualizada

La impresión humana mejora en un aspecto y empeora en otro.

## Mejora

EXP-04 muestra que el candidato **sí sabe hacer una comparación iso-parámetro limpia**:

- congela el modelo;
- iguala presupuestos;
- cuenta parámetros entrenables;
- comprueba merge;
- compara LoRA vs SpecRAMA en condiciones razonables.

Eso es buena señal.

## Empeora

Los experimentos posteriores parecen haber mezclado piezas de protocolos distintos:

- tamaños de EXP-04;
- targets distintos en EXP-11;
- LoRA no congelado en EXP-11/12;
- subset de test presentado como test completo;
- fake NF4 presentado como formato 4-bit;
- adapter size sin buffers.

Eso es una señal de **falta de control experimental**, no de falta de capacidad.

---

# Veredicto de reclutamiento

Mi decisión sigue siendo:

```text
Entrevista técnica: sí.
Oferta directa: no.
Reproducción controlada: obligatoria.
```

Pero ahora sería más específico:

> El candidato tiene inventiva real y sabe escribir código de investigación. EXP-04 demuestra capacidad para diseñar comparaciones justas. El problema es que necesita disciplina de benchmark: una única configuración canónica, asserts de parámetros, splits correctos, seeds, logs crudos y lenguaje más cuidadoso.

Puntuación actualizada:

| Dimensión | Evaluación |
|---|---:|
| Originalidad | 8.5/10 |
| Capacidad de implementación | 7.5/10 |
| Capacidad de diseño experimental | 6.5/10 |
| Consistencia experimental actual | 5/10 |
| Honestidad intelectual | 8/10 |
| Potencial como research engineer | 8/10 |

---

# Siguiente archivo que quiero leer

Ahora quiero ver:

```text
spec_rama/shared_layers.py
```

Motivo: EXP-04 incluye el claim de:

```text
SpecRAMA Shared (Sub-KB)
```

y necesito auditar si el modelo compartido:

- congela correctamente;
- comparte cores como se afirma;
- cuenta bien los parámetros;
- mantiene merge correcto;
- no introduce parámetros ocultos o buffers grandes.

Si además tienes el output/log bruto de EXP-04, también sería útil, pero el siguiente archivo prioritario es:

```text
spec_rama/shared_layers.py
```

---


# Auditoría de `shared_layers.py`

Este archivo revela un problema importante: la afirmación **“Sub-Kilobyte PEFT” sólo es cierta para los parámetros entrenables ideales**, no para el adaptador tal como está representado por la implementación actual.

Además, el código contiene parámetros locales que quedan registrados, entrenables y sin uso efectivo.

## Hallazgo crítico: los núcleos locales no se eliminan

`SharedSpecRAMALinear` crea primero una capa normal:

```python
self.spec_layer = SpecRAMALinear(
    ...,
    use_multiplicative=(master_core_m is not None),
    use_additive=(master_core_a is not None),
)
```

Eso crea dentro de `spec_layer`:

```python
spec_layer.core_m
spec_layer.core_a
```

Después se guardan los núcleos compartidos:

```python
self.master_core_m = master_core_m
self.master_core_a = master_core_a
```

Pero los núcleos locales no se reemplazan ni se desregistran.

En `get_effective_weight()` se utilizan los núcleos compartidos:

```python
core_m_scaled = self.master_core_m * self.gamma_m
core_a_scaled = self.master_core_a * self.gamma_a
```

y se pasan como `custom_core_m` y `custom_core_a`. Por tanto:

- `spec_layer.core_m` y `spec_layer.core_a` permanecen como parámetros;
- aparecen en `model.parameters()`;
- el optimizador los recibe;
- pero no influyen en el resultado;
- normalmente no reciben gradiente útil.

Son parámetros muertos desde el punto de vista funcional.

## Conteo real en EXP-04

EXP-04 utiliza:

```python
targets = ["c_attn", "c_proj"]
```

Con el selector por subcadena, esto envuelve por bloque:

```text
attn.c_attn
attn.c_proj
mlp.c_proj
```

GPT-2 Small tiene 12 bloques, por lo que se crean 36 capas compartidas.

Con `core_size=(8, 8)` y usando componentes multiplicativa y aditiva:

### Parámetros esperados en una implementación corregida

Dos núcleos maestros:

\[
2 \times 8 \times 8 = 128
\]

Dos ganancias por capa:

\[
2 \times 36 = 72
\]

Total funcional:

\[
128 + 72 = 200\text{ parámetros}
\]

En FP32:

\[
200 \times 4 = 800\text{ bytes}
\]

Eso sí es aproximadamente sub-KB, excluyendo buffers y metadatos.

### Parámetros del código actual

Cada capa compartida conserva además dos núcleos locales:

\[
2 \times 8 \times 8 = 128
\]

Para 36 capas:

\[
36 \times 128 = 4.608
\]

Añadiendo núcleos maestros y ganancias:

\[
4.608 + 128 + 72 = 4.808\text{ parámetros}
\]

En FP32:

\[
4.808 \times 4 = 19.232\text{ bytes}
\]

Aproximadamente:

- 18,8 KiB;
- 19,2 KB decimales.

Por tanto, si se ejecuta exactamente el código suministrado, la línea:

```text
SpecRAMA Shared (Sub-KB)
```

no debería reportar un adaptador sub-KB. Debería contar alrededor de **4.808 parámetros entrenables**, aunque 4.608 de ellos sean parámetros inútiles.

## Corrección mínima

Después de construir `SpecRAMALinear`, habría que eliminar los núcleos locales:

```python
self.spec_layer.register_parameter("core_m", None)
self.spec_layer.register_parameter("core_a", None)
```

Una solución mejor sería modificar `SpecRAMALinear` para aceptar núcleos externos y no crear parámetros locales en absoluto.

Por ejemplo:

```python
self.spec_layer = SpecRAMALinear(
    base_layer=base_layer,
    transform_type=transform_type,
    core_size=core_size,
    alpha_m=alpha_m,
    alpha_a=alpha_a,
    permutation_method=permutation_method,
    use_multiplicative=False,
    use_additive=False,
)
```

Pero entonces `get_effective_weight()` tendría que aceptar explícitamente las componentes externas sin depender de `self.use_multiplicative` y `self.use_additive`.

La arquitectura más limpia sería:

```python
SpecRAMALinear(
    base_layer=base_layer,
    external_core_m=master_core_m,
    external_core_a=master_core_a,
    ...
)
```

## El comentario de “un único master core” es incorrecto

La clase crea:

```python
self.master_core_m = nn.Parameter(torch.zeros(*core_size))
self.master_core_a = nn.Parameter(torch.zeros(*core_size))
```

Hay dos núcleos maestros:

- uno multiplicativo;
- uno aditivo.

Por tanto, la documentación debería decir:

> Dos núcleos maestros compartidos más dos ganancias escalares por capa.

Si se pretendiera realmente usar un único núcleo, habría que especificar cómo se reutiliza entre ambas ramas.

## Sobre el registro de los núcleos compartidos

Al hacer:

```python
self.master_core_m = master_core_m
```

dentro de cada `SharedSpecRAMALinear`, PyTorch registra la misma instancia de `Parameter` tanto en el modelo raíz como en cada wrapper.

`model.parameters()` normalmente deduplica parámetros compartidos por identidad, así que el conteo no debería multiplicar los núcleos maestros. Sin embargo, el `state_dict()` puede contener varias rutas apuntando al mismo parámetro, por ejemplo:

```text
master_core_m
model.transformer...master_core_m
model.transformer...master_core_m
...
```

Esto complica:

- serialización;
- carga de checkpoints;
- inspección del tamaño;
- compatibilidad con herramientas PEFT;
- gestión de claves duplicadas.

Sería preferible que los núcleos maestros vivieran únicamente en `SharedSpecRAMAModel` y que los wrappers mantuvieran una referencia no registrada, o que utilizaran un módulo explícito de almacenamiento compartido.

## El adaptador completo no es sub-KB

Aunque el número de parámetros funcionales pueda ser 200, cada capa conserva cuatro permutaciones:

```python
row_perm
col_perm
row_inv_perm
col_inv_perm
```

Son tensores `torch.long`, normalmente de 8 bytes por índice.

Para los 36 módulos de EXP-04, el coste aproximado de las permutaciones es:

```text
≈ 1.622.016 bytes
```

Con wavelets no se registran las matrices DCT/Walsh, pero las permutaciones siguen siendo necesarias para reconstruir cada actualización.

Por tanto, hay que distinguir:

| Medida | Valor aproximado |
|---|---:|
| Parámetros funcionales corregidos | 200 |
| Núcleos funcionales en FP32 | 800 bytes |
| Parámetros actuales incluyendo núcleos muertos | 4.808 |
| Núcleos actuales en FP32 | 19.232 bytes |
| Permutaciones por capa | ≈1,62 MB |
| Adaptador completo actual | superior a 1,6 MB |

La afirmación correcta sería:

> Spec-RAMA puede tener un número sub-KB de parámetros entrenables compartidos, pero la implementación actual almacena además permutaciones por capa que dominan el tamaño del adaptador.

Sólo podría defenderse un adaptador físico sub-KB si las permutaciones se:

- regeneran determinísticamente desde el modelo base;
- excluyen del checkpoint;
- o se comprimen con una representación compacta.

## El modelo compartido no congela el modelo por sí mismo

`SharedSpecRAMAModel` no ejecuta:

```python
for p in model.parameters():
    p.requires_grad = False
```

Por tanto, utilizado de forma independiente sobre un modelo no congelado, podría dejar entrenables:

- embeddings;
- LayerNorm;
- capas no objetivo;
- otros parámetros del modelo.

En EXP-04 sí se congela antes:

```python
for p in model.parameters():
    p.requires_grad = False
```

y los nuevos núcleos y ganancias se crean después con `requires_grad=True`. En ese experimento concreto, el orden es correcto salvo por los núcleos locales muertos.

La clase debería ser segura por defecto y congelar explícitamente el modelo base o verificar que sólo están entrenables:

```python
master_core_m
master_core_a
gamma_m
gamma_a
```

## Problema crítico en `merge()`: no es idempotente

El método implementado es:

```python
def merge(self):
    w_eff = self.get_effective_weight()
    self.spec_layer.base_layer.weight.data.copy_(w_eff)
    self.spec_layer.merged = True
```

No comienza con:

```python
if self.spec_layer.merged:
    return
```

Esto permite aplicar la actualización dos veces.

Primera llamada:

\[
W \leftarrow W_0 + \Delta W
\]

Segunda llamada:

\[
W \leftarrow (W_0 + \Delta W) + \Delta W
= W_0 + 2\Delta W
\]

La implementación de `SpecRAMALinear.merge()` sí tiene una guardia de idempotencia, pero `SharedSpecRAMALinear.merge()` no.

La corrección mínima:

```python
def merge(self):
    if self.spec_layer.merged:
        return

    with torch.no_grad():
        w_eff = self.get_effective_weight()
        self.spec_layer.base_layer.weight.copy_(w_eff)

    self.spec_layer.merged = True
```

También debería guardarse el peso original para poder implementar `unmerge()`.

## Problema de persistencia del estado fusionado

`merged` es un atributo Python normal:

```python
self.spec_layer.merged = True
```

No es un parámetro ni un buffer. Por tanto, no se guarda en el `state_dict`.

El flujo siguiente puede aplicar la adaptación dos veces:

1. se fusiona el modelo;
2. se guarda el checkpoint;
3. se carga en un nuevo modelo;
4. `merged` vuelve a `False`;
5. el wrapper vuelve a añadir la actualización compartida sobre los pesos ya fusionados.

Este problema ya estaba presente en la capa normal, pero aquí es particularmente delicado porque `SharedSpecRAMALinear` no ofrece `unmerge()`.

La solución preferible es una de estas:

- guardar un buffer persistente `merged`;
- eliminar los wrappers después de fusionar;
- almacenar checkpoints explícitamente como “unmerged” o “merged”;
- implementar una rutina de carga que establezca correctamente el estado.

## EXP-04 no prueba realmente la fusión del modelo compartido

En `run_exp04`:

```python
if method in ["wavelet", "dct"]:
    merge_spec_rama_modules(model)
    merged_ppl, _ = evaluate_on_dataset(model, test_data)
elif method == "lora":
    ...
else:
    merged_ppl = test_ppl
```

Para:

```python
method == "shared"
```

se asigna:

```python
merged_ppl = test_ppl
```

No se llama a:

```python
model.merge()
```

Por tanto, la fila de Shared Spec-RAMA no tiene una medición real post-merge. El valor `Merged PPL` es simplemente una copia del PPL sin fusionar.

Esto debe corregirse:

```python
if method == "shared":
    model.merge()
    merged_ppl, _ = evaluate_on_dataset(model, test_data)
```

Y luego debe verificarse la equivalencia numérica entre ambos caminos.

## Problema de selección de módulos

La clase utiliza otra vez:

```python
any(t in name for t in self.target_modules)
```

Con:

```python
targets = ["c_attn", "c_proj"]
```

se seleccionan:

```text
attn.c_attn
attn.c_proj
mlp.c_proj
```

La selección debe basarse en nombres completos, no en subcadenas locales. De lo contrario, el resultado depende accidentalmente de que dos módulos distintos compartan el sufijo `c_proj`.

## Evaluación revisada

| Dimensión | Valoración |
|---|---:|
| Idea de compartir núcleos | 8/10 |
| Implementación de la parametrización | 5/10 |
| Gestión de parámetros compartidos | 4/10 |
| Conteo de tamaño | 3/10 |
| Fusión y persistencia | 3/10 |
| Potencial de investigación | 8/10 |

# Veredicto editorial

La idea de compartir un núcleo espectral entre capas es interesante y podría ser una contribución sólida:

- reduce la dimensionalidad efectiva;
- fuerza una estructura común entre capas;
- permite estudiar el compromiso entre compartir y especializar;
- ofrece un régimen de adaptación extremadamente compacto.

Pero la afirmación actual debe reformularse:

> La implementación demuestra una parametrización compartida con potencial sub-KB en parámetros funcionales, no un adaptador sub-KB en memoria ni un sistema de despliegue sub-KB.

Además, el resultado de EXP-04 debe marcarse como pendiente de auditoría porque:

- el conteo incluye núcleos locales no utilizados;
- el `Merged PPL` de la variante shared no se mide realmente;
- las permutaciones dominan el tamaño físico;
- la persistencia de `merged` no está resuelta.

## Recomendación al candidato

Le pediría una corrección mínima antes de aceptar la afirmación sub-KB:

1. eliminar los núcleos locales de `SpecRAMALinear`;
2. conservar sólo dos núcleos maestros y las ganancias;
3. imprimir nombres y gradientes de todos los parámetros entrenables;
4. medir el `state_dict()` completo del adaptador;
5. distinguir parámetros funcionales de buffers;
6. hacer `merge()` idempotente;
7. añadir `unmerge()` y pruebas de save/load;
8. repetir EXP-04 después de medir realmente el camino fusionado.

La recomendación de contratación se mantiene, pero el candidato debe demostrar que puede transformar prototipos creativos en artefactos experimentales confiables. Esa transición es precisamente la diferencia entre una buena idea y una contribución de laboratorio.

El siguiente archivo que quiero revisar es:

```text
spec-rama/tests/test_shared_layers.py
```

Si no existe, envíame el log completo de `EXP-04`, especialmente las líneas `Trainable`, `TEST PPL` y `Merged PPL` de `SpecRAMA Shared`.

---


## Auditoría de `shared_layers.py`

Este archivo confirma otra inconsistencia importante: el concepto de **Shared SpecRAMA sub-KB** es interesante, pero la implementación actual probablemente **no implementa el conteo sub-kilobyte de forma limpia**.

La idea es buena:

> Un único núcleo espectral maestro compartido por muchas capas + ganancias escalares por capa.

Pero el código actual introduce parámetros locales no usados y posiblemente duplica el registro de los núcleos maestros.

---

# Hallazgo crítico: se crean cores locales no usados

En `SharedSpecRAMALinear.__init__()` haces:

```python
self.spec_layer = SpecRAMALinear(
    base_layer=base_layer,
    transform_type=transform_type,
    core_size=core_size,
    alpha_m=alpha_m,
    alpha_a=alpha_a,
    permutation_method=permutation_method,
    use_multiplicative=(master_core_m is not None),
    use_additive=(master_core_a is not None),
)
```

Pero `SpecRAMALinear` crea internamente:

```python
self.core_m = nn.Parameter(...)
self.core_a = nn.Parameter(...)
```

Luego en `SharedSpecRAMALinear` dices:

```python
self.master_core_m = master_core_m
self.master_core_a = master_core_a
```

y en forward usas:

```python
core_m_scaled = self.master_core_m * self.gamma_m
core_a_scaled = self.master_core_a * self.gamma_a
```

Es decir:

```text
Los core_m/core_a internos de self.spec_layer existen, requieren gradiente, aparecen en parameters(), pero no se usan en forward.
```

Esto rompe el claim limpio de:

```text
Total trainable parameters = 1 master core + 2 * number of layers
```

En realidad, con el código actual, también existen:

```text
core_m/core_a locales por cada capa envuelta
```

aunque no contribuyan al cálculo.

---

# Conteo esperado vs conteo real

Para GPT-2 Small con:

```python
targets = ["c_attn", "c_proj"]
```

se envuelven 36 módulos:

```text
12 capas × {attn.c_attn, attn.c_proj, mlp.c_proj} = 36 módulos
```

Con core `8×8` y dos vías, multiplicativa + aditiva:

## Conteo ideal del modelo compartido

```text
master_core_m: 8×8 = 64
master_core_a: 8×8 = 64
gammas: 2 × 36 = 72

total ideal = 200 parámetros
bytes FP32 = 800 bytes
```

Eso sí sería sub-KB.

## Conteo probable con el código actual

Además de esos 200 parámetros, cada `SpecRAMALinear` interno crea:

```text
core_m local: 64
core_a local: 64
total local por módulo = 128
```

Para 36 módulos:

```text
36 × 128 = 4.608 parámetros locales no usados
```

Total probable:

```text
4.608 + 200 = 4.808 parámetros
bytes FP32 = 19.232 bytes ≈ 18.8 KiB
```

Eso ya no es sub-KB.

Sigue siendo pequeño, pero no coincide con el claim.

---

# Posible duplicación de registro de parámetros maestros

También hay otra sutileza de PyTorch:

```python
self.master_core_m = master_core_m
self.master_core_a = master_core_a
```

Como `master_core_m` y `master_core_a` son `nn.Parameter`, al asignarlos como atributos de cada `SharedSpecRAMALinear`, PyTorch puede registrarlos también como parámetros del submódulo.

Puede que `named_parameters(remove_duplicate=True)` deduplique objetos compartidos, pero el `state_dict` puede acabar con claves duplicadas o confusas.

Más limpio sería que los submódulos no registren el parámetro maestro, sino que guarden una referencia no registrada o que reciban el core explícitamente desde el módulo padre.

---

# `SharedSpecRAMAModel` tampoco congela el modelo base

En EXP-04 esto no rompe el resultado porque antes haces:

```python
for p in model.parameters():
    p.requires_grad = False
```

Pero la clase en sí no lo garantiza.

Si alguien usa directamente:

```python
model = SharedSpecRAMAModel(model, ...)
```

sin congelar antes, entrenará también parámetros base.

Para una API PEFT segura, debería haber:

```python
freeze_base=True
```

por defecto.

---

# `merge()` puede aplicar doble actualización si se llama dos veces

En `SharedSpecRAMALinear.merge()`:

```python
def merge(self):
    w_eff = self.get_effective_weight()
    self.spec_layer.base_layer.weight.data.copy_(w_eff)
    self.spec_layer.merged = True
```

No hay guard:

```python
if self.spec_layer.merged:
    return
```

Si `merge()` se llama dos veces, la segunda vez puede calcular `w_eff` sobre un peso base ya modificado y aplicar la adaptación de nuevo.

`SpecRAMALinear.merge()` sí tiene guard. `SharedSpecRAMALinear.merge()` debería tenerlo también.

---

# Evaluación del claim “Sub-KB”

La idea es defendible, pero el código actual no la implementa de forma limpia.

Yo lo calificaría así:

```text
Concepto: fuerte.
Implementación actual: prototipo con bug de conteo.
Claim sub-KB: no defendible hasta corregir parámetros locales no usados y estado serializado.
```

La versión correcta debería reportar:

```text
Active trainable params: 200
Serialized adapter size incluyendo permutaciones: mayor
Full state_dict delta: medir empíricamente
```

---

# Parche mínimo recomendado

## 1. Evitar que los cores locales de `SpecRAMALinear` cuenten como entrenables

Después de crear `self.spec_layer`, hacer algo como:

```python
if self.spec_layer.core_m is not None:
    self.spec_layer.core_m.requires_grad = False

if self.spec_layer.core_a is not None:
    self.spec_layer.core_a.requires_grad = False
```

Eso arregla el conteo de `requires_grad`, aunque no elimina esos tensores del `state_dict`.

Mejor aún sería eliminar esos parámetros o refactorizar `SpecRAMALinear` para aceptar cores externos sin crear cores locales.

---

## 2. Añadir guard a `merge()`

```python
def merge(self):
    if self.spec_layer.merged:
        return

    w_eff = self.get_effective_weight()
    with torch.no_grad():
        self.spec_layer.base_layer.weight.copy_(w_eff)
    self.spec_layer.merged = True
```

---

## 3. Congelar el modelo dentro de `SharedSpecRAMAModel`

```python
for p in model.parameters():
    p.requires_grad = False
```

antes de inyectar.

---

## 4. Evitar registrar el master core en cada subcapa

En vez de:

```python
self.master_core_m = master_core_m
```

usar una referencia no registrada o pasar los cores explícitamente desde el padre. Por ejemplo:

```python
object.__setattr__(self, "_master_core_m", master_core_m)
object.__setattr__(self, "_master_core_a", master_core_a)
```

y luego:

```python
core_m_scaled = self._master_core_m * self.gamma_m
core_a_scaled = self._master_core_a * self.gamma_a
```

Así los parámetros maestros quedan registrados solo en `SharedSpecRAMAModel`.

---

# Veredicto actualizado

Mi evaluación del candidato se mantiene:

> **Muy creativo, buen prototipador, pero con problemas repetidos de higiene experimental y contabilidad de parámetros.**

Este archivo refuerza el patrón visto antes:

- la idea es buena;
- el código funciona como prototipo;
- los claims se adelantan a la implementación;
- falta una pasada dura de auditoría antes de publicar resultados.

Como reclutador, todavía diría:

```text
Entrevista técnica: sí.
Oferta directa: no.
Prueba de reproducción limpia: obligatoria.
```

El candidato tiene potencial claro para un rol de **research engineer en efficient ML / PEFT**, pero necesitaría demostrar que puede convertir prototipos creativos en benchmarks canónicos, reproducibles y cuidadosamente auditados.

---

# Siguiente cosa que quiero ver

Ahora no quiero otro archivo de implementación. Quiero ver el output real.

Copia, si lo tienes:

```text
log/output bruto de benchmarks/exp04_equal_parameter_head2head.py
```

Especialmente las líneas donde imprime:

```text
Trainable Params
Adapter Size
TEST PPL
Merged PPL
```

Motivo: necesitamos comprobar si el experimento realmente imprimió el tamaño sub-KB o si el propio `count_trainable_parameters()` ya mostraba ~18.8 KiB para `SharedSpecRAMA`.

Si no tienes el log bruto, entonces copia:

```text
docs/findings_spec_rama_v4.md
```


---

# Auditoría de EXP-04: la tabla permite verificar los parámetros, pero no la memoria ni el alcance de la evaluación

El hecho de que no aparezca el output bruto impide verificar directamente los valores de PPL, pero la tabla v4 permite auditar varias propiedades estáticas. De hecho, sus conteos de parámetros coinciden casi exactamente con el código entregado.

## Corrección de mi evaluación anterior

Debo corregir una parte de mi análisis previo:

> Para EXP-04, los conteos publicados sí son compatibles con el código actual.

La configuración:

```python
targets = ["c_attn", "c_proj"]
```

selecciona, mediante coincidencia por subcadena:

```text
attn.c_attn
attn.c_proj
mlp.c_proj
```

Es decir, tres módulos por bloque y 36 módulos en GPT-2 Small.

Esto explica todos los conteos de la tabla:

| Configuración | Cálculo | Resultado |
|---|---:|---:|
| Spec-RAMA 8×8 | \(36 \times 2 \times 8 \times 8\) | 4.608 |
| LoRA rank 1 | \(36 \times (d_\text{in}+d_\text{out})\) | 101.376 |
| Spec-RAMA 37×38 | \(36 \times 2 \times 37 \times 38\) | 101.232 |
| LoRA rank 4 | \(4 \times 101.376\) | 405.504 |
| Spec-RAMA 75×75 | \(36 \times 2 \times 75 \times 75\) | 405.000 |

Por tanto, la tabla parece haber sido generada con una versión compatible con los scripts mostrados, al menos en lo referente al conteo de parámetros.

La discrepancia de módulos que señalé anteriormente sigue afectando a EXP-11/EXP-12, donde se usa:

```python
targets = ["c_attn", "c_proj", "c_fc"]
```

y el selector actual incluiría cuatro módulos por bloque. Pero no afecta de la misma manera a EXP-04, que utiliza solamente `c_attn` y `c_proj`.

# 1. “Iso-Parameter” es aproximado, no exacto

Los presupuestos están muy bien igualados, pero no son idénticos:

- LoRA rank 1: 101.376 parámetros;
- Spec-RAMA 37×38: 101.232 parámetros;
- diferencia: 144 parámetros, aproximadamente 0,14%.

Y:

- LoRA rank 4: 405.504 parámetros;
- Spec-RAMA 75×75: 405.000 parámetros;
- diferencia: 504 parámetros, aproximadamente 0,12%.

La formulación correcta sería:

> Comparación con presupuesto de parámetros prácticamente igualado.

“Exact iso-parameter” es demasiado fuerte, aunque la diferencia práctica sea insignificante.

# 2. El resultado más importante de EXP-04 es negativo para Spec-RAMA

Los números muestran una ventaja clara de LoRA en calidad:

| Presupuesto aproximado | LoRA | Spec-RAMA Wavelet | Diferencia |
|---:|---:|---:|---:|
| ~101K | 30,57 PPL | 37,81 PPL | +7,24 PPL |
| ~405K | 29,67 PPL | 35,90 PPL | +6,23 PPL |

La variante DCT obtiene 38,41 PPL frente a 30,57 PPL de LoRA en el presupuesto de ~101K.

Por tanto, EXP-04 no demuestra que Spec-RAMA sea competitivo con LoRA a igualdad de parámetros. Demuestra algo más matizado:

- en el régimen inferior a 20 KB, Spec-RAMA mejora el modelo base;
- cuando LoRA puede utilizar 100K–400K parámetros, LoRA obtiene resultados sustancialmente mejores;
- la representación espectral tiene una posible ventaja de compactación extrema, pero no de calidad por parámetro en este experimento.

La conclusión científicamente honesta sería:

> Spec-RAMA ofrece adaptación útil en presupuestos muy pequeños, pero queda claramente por debajo de LoRA cuando ambos disponen de aproximadamente 100K o 400K parámetros.

Esto no invalida la línea de investigación. Sí invalida cualquier afirmación de superioridad general frente a LoRA.

## Lectura de la variante compartida

La variante shared obtiene:

```text
GPT-2 base: 46,18 PPL
Shared Spec-RAMA: 45,54 PPL
```

La mejora es pequeña, pero es destacable porque utiliza una parametrización funcional extremadamente reducida. La variante wavelet no compartida obtiene 42,57 PPL con 4.608 parámetros, por lo que compartir un único núcleo parece imponer una limitación fuerte.

Esto sugiere una pregunta de investigación interesante:

> ¿Cuánta especialización por capa necesita un núcleo espectral compartido para conservar capacidad?

# 3. La etiqueta “Sub-KB” es incorrecta en el código actual

La tabla declara:

```text
SpecRAMA Shared (Sub-KB)
4.808 parámetros
18,8 KB
```

Esto es internamente contradictorio:

- 4.808 parámetros float32 ocupan aproximadamente 18,8 KiB;
- no son menos de 1 KB;
- por tanto, la configuración implementada no es sub-KB.

La razón es el defecto ya identificado en `SharedSpecRAMALinear`.

Cada wrapper crea núcleos locales:

```python
self.spec_layer = SpecRAMALinear(...)
```

y esos núcleos contienen:

```python
core_m: 8 × 8
core_a: 8 × 8
```

Por cada una de las 36 capas:

\[
2 \times 8 \times 8 = 128
\]

parámetros locales muertos.

Total de parámetros locales:

\[
36 \times 128 = 4.608
\]

Además se crean:

- dos núcleos maestros: 128 parámetros;
- dos ganancias por capa: \(36 \times 2=72\).

Total:

\[
4.608+128+72=4.808
\]

La tabla, por tanto, confirma el defecto de implementación.

## Tamaño funcional real

Los núcleos locales no se utilizan en:

```python
get_effective_weight()
```

porque se pasan los núcleos maestros mediante:

```python
custom_core_m=core_m_scaled
custom_core_a=core_a_scaled
```

La parametrización funcional es solamente:

\[
128+72=200\text{ parámetros}
\]

En float32:

```text
200 × 4 = 800 bytes
```

Así que la afirmación sub-KB sería defendible **después de eliminar los 4.608 parámetros locales inutilizados**. El resultado actual de 45,54 PPL parece provenir funcionalmente de esos aproximadamente 200 parámetros, no de 4.808 parámetros efectivos.

La tabla debería mostrar ambas cifras:

| Métrica | Valor |
|---|---:|
| Parámetros registrados actuales | 4.808 |
| Parámetros con influencia funcional | 200 |
| Tamaño funcional FP32 | 800 bytes |

# 4. La memoria total publicada no es la memoria física del modelo

La tabla calcula el tamaño del adaptador como:

```python
size_bytes = trainable * 4
```

Eso sólo cuenta parámetros entrenables en float32. No incluye buffers ni memoria temporal.

## Wavelet

Cada `SpecRAMALinear` registra:

```python
row_perm
col_perm
row_inv_perm
col_inv_perm
```

Son tensores `torch.long`.

Para los 36 módulos de EXP-04, las permutaciones ocupan aproximadamente:

```text
1.622.016 bytes ≈ 1,55 MiB
```

Esto es mucho mayor que el adaptador wavelet de 8×8:

```text
4.608 parámetros × 4 bytes = 18.432 bytes
```

Por tanto, el modelo wavelet de 8×8 no añade físicamente sólo 18,0 KB. Añade, aproximadamente:

```text
18 KB de núcleos
+ 1,55 MiB de permutaciones
```

La misma observación afecta a las variantes 37×38, 75×75 y shared.

## DCT

La discrepancia es mucho más grave en DCT. `SpecRAMALinear` registra matrices DCT completas:

```python
self.register_buffer("B_out", get_dct_matrix_1d(self.out_features))
self.register_buffer("B_in", get_dct_matrix_1d(self.in_features))
```

Para los 36 módulos seleccionados de GPT-2 Small, esas bases ocupan aproximadamente:

```text
783 MiB en float32
```

Esto se debe a que se almacenan matrices densas de dimensiones:

- 2304×2304;
- 768×768;
- 3072×3072;

para cada capa correspondiente.

Por tanto, la fila:

```text
SpecRAMA DCT (37x38) | Tamaño Total Modelo: 475,1 MB
```

no representa la memoria real de la implementación actual. El modelo DCT debería estar aproximadamente en el orden de:

```text
474,7 MiB de pesos base
+ 783 MiB de bases DCT
+ permutaciones
+ núcleos
```

es decir, alrededor de **1,26 GiB**, antes de considerar estados del optimizador o activaciones.

La tabla sólo puede interpretarse como:

> tamaño del modelo base más parámetros entrenables, ignorando buffers de transformación y permutaciones.

No como memoria total de ejecución.

# 5. EXP-04 no verifica el merge de la variante shared

El código contiene:

```python
if method in ["wavelet", "dct"]:
    merge_spec_rama_modules(model)
    merged_ppl, _ = evaluate_on_dataset(model, test_data)
elif method == "lora":
    ...
else:
    merged_ppl = test_ppl
```

Para:

```python
method == "shared"
```

no se llama a:

```python
model.merge()
```

Así que la variante shared no tiene una medición post-merge real. En caso de que el documento sólo publique `TEST PPL`, como ocurre aquí, el problema no afecta al número mostrado, pero sí significa que no existe evidencia de equivalencia tras fusionar.

Además, `SharedSpecRAMALinear.merge()` no es idempotente. Si se llama dos veces, puede aplicar la actualización dos veces:

\[
W_0+\Delta W \rightarrow W_0+2\Delta W
\]

Ese método debe corregirse antes de considerar validada la ruta de despliegue.

# 6. El test set sigue sin ser de 280.000 tokens según el código

El benchmark llama a:

```python
prepare_wikitext_data(
    block_size=256,
    max_train_samples=400,
    max_test_samples=100
)
```

Por tanto, el código evalúa como máximo:

```text
100 bloques × 256 tokens = 25.600 tokens de entrada
```

La evaluación causal efectiva es ligeramente menor porque el modelo desplaza las etiquetas internamente.

La documentación afirma:

```text
WikiText-2 Test Set (280.000 tokens)
```

pero eso no coincide con el script suministrado.

Sin el output bruto no puedo determinar si:

- la tabla procede de otra versión del script que usaba todo el test;
- la documentación exagera el protocolo;
- o el resultado fue generado con una configuración distinta.

Los conteos de parámetros sí coinciden con el código actual; los PPL y el número de tokens evaluados no pueden verificarse de la misma manera.

# 7. Qué conclusiones sí son defendibles

Con el material disponible, publicaría EXP-04 así:

> En un benchmark exploratorio sobre GPT-2 Small y un subconjunto de WikiText-2, Spec-RAMA mejora el modelo base con menos de 20 KB de parámetros entrenables. Sin embargo, a presupuestos de aproximadamente 100K y 400K parámetros, LoRA obtiene una perplejidad considerablemente menor. El resultado sugiere que la ventaja de Spec-RAMA está en la adaptación extremadamente compacta, no en superar a LoRA bajo presupuestos equivalentes.

No publicaría:

- “igualdad exacta”;
- “memoria total del modelo”;
- “DCT con 475,1 MB”;
- “Shared Sub-KB” para la implementación actual;
- “Spec-RAMA supera a LoRA”;
- “evaluación sobre 280.000 tokens” sin corregir el script.

# Evaluación del candidato tras EXP-04

Este documento mejora la evaluación en un aspecto importante: el candidato parece estar dispuesto a publicar resultados donde Spec-RAMA pierde claramente contra LoRA. Eso es una señal positiva. La tabla no intenta ocultar que:

```text
LoRA rank=1: 30,57 PPL
Spec-RAMA 37×38: 37,81 PPL
```

y:

```text
LoRA rank=4: 29,67 PPL
Spec-RAMA 75×75: 35,90 PPL
```

La investigación parece tener una hipótesis más limitada y más interesante:

> Las bases espectrales pueden ser útiles cuando el presupuesto de adaptación es tan pequeño que el LoRA convencional deja de ser viable.

Esa hipótesis merece ser probada contra métodos realmente compactos como VeRA, NOLA, FourierFT y variantes de LoRA compartido, no sólo contra LoRA estándar.

## Valoración actualizada

| Área | Puntuación |
|---|---:|
| Originalidad | 8,5/10 |
| Capacidad de prototipado | 8/10 |
| Transparencia al mostrar resultados negativos | 7,5/10 |
| Corrección de las métricas de memoria | 3/10 |
| Rigor del protocolo de evaluación | 4/10 |
| Ingeniería de despliegue | 3,5/10 |
| Potencial como Research Engineer | 8/10 |

## Veredicto

**Mantengo la recomendación de entrevista técnica.**

La entrevista debería centrarse en una pregunta fundamental:

> ¿Puede el candidato distinguir y medir correctamente parámetros entrenables, tamaño serializado, memoria de ejecución, coste de entrenamiento y coste de inferencia?

Si responde bien a esa pregunta y corrige el sistema, el proyecto puede convertirse en una línea de investigación valiosa. Si no, existe el riesgo de que las contribuciones se basen en mejoras aparentes derivadas de definiciones incompletas de tamaño o protocolo.

El siguiente archivo que quiero leer es:

```text
spec-rama/tests/test_layers.py
```

Si no existe, envíame cualquier archivo de tests relacionado con:

```text
merge
shared
state_dict
permutation
```

Necesito comprobar si hay pruebas que garanticen equivalencia antes/después de fusionar y que detecten los parámetros locales muertos de `SharedSpecRAMALinear`.

---

## Evaluación de `findings_spec_rama_v4.md`

Este documento confirma varias cosas importantes.

La buena noticia: **EXP-04 es probablemente el experimento más honesto de SpecRAMA hasta ahora**, porque sí reporta conteos reales de parámetros entrenables.  
La mala noticia: **los propios resultados muestran que LoRA domina claramente a SpecRAMA en régimen iso-parámetro**.

---

# 1. Hallazgo clave: el claim “Sub-KB” queda refutado por la propia tabla

La tabla dice:

```text
SpecRAMA Shared (Sub-KB): 4,808 parámetros, 18.8 KB
```

Eso coincide exactamente con el problema que habíamos detectado en `shared_layers.py`.

El ideal teórico era:

```text
master_core_m: 64
master_core_a: 64
gammas: 72
total ideal: 200 params ≈ 800 bytes
```

Pero la implementación real cuenta:

```text
4,808 params ≈ 18.8 KB
```

porque cada `SharedSpecRAMALinear` crea cores locales no usados.

Por tanto, la etiqueta correcta no es:

```text
SpecRAMA Shared (Sub-KB)
```

sino:

```text
SpecRAMA Shared (~19 KB, implementation with unused local cores)
```

La tabla es honesta; el nombre no.

---

# 2. Resultado científico más importante: LoRA gana claramente a igualdad de parámetros

La tabla muestra:

| Régimen | LoRA | SpecRAMA | Diferencia |
|---|---:|---:|---:|
| ~100K params | LoRA r=1: **30.57 PPL** | SpecRAMA 37x38: **37.81 PPL** | LoRA mucho mejor |
| ~405K params | LoRA r=4: **29.67 PPL** | SpecRAMA 75x75: **35.90 PPL** | LoRA mucho mejor |

Esto es una conclusión fuerte:

> En EXP-04, SpecRAMA no es competitivo con LoRA en calidad por parámetro cuando ambos tienen el mismo presupuesto.

Esto no invalida SpecRAMA, pero sí obliga a cambiar la narrativa.

La narrativa correcta sería:

> SpecRAMA ofrece una familia de adaptadores espectrales extremadamente compactos que pueden mejorar el modelo en presupuestos donde LoRA full-target no existe o no es práctico. Pero, en presupuestos comparables, LoRA espacial sigue siendo más fuerte en PPL.

No diría:

> “SpecRAMA supera o iguala a LoRA iso-parámetro.”

Los datos no apoyan eso.

---

# 3. El nicho real de SpecRAMA parece ser ultra-low-budget, no iso-parámetro

Los resultados de menor tamaño son interesantes:

```text
GPT-2 Base: 46.18 PPL
SpecRAMA Shared: 45.54 PPL, 18.8 KB
SpecRAMA Wavelet 8x8: 42.57 PPL, 18.0 KB
```

Esto sí es valioso.

Una mejora de:

```text
46.18 → 42.57 PPL
```

con solo:

```text
4,608 parámetros entrenables
```

es una señal interesante.

Pero hay que compararlo contra baselines ultra-compactos reales:

- IA³;
- VeRA;
- BitFit;
- LoRA aplicado solo a una o dos capas;
- LoRA solo en `lm_head` si no está atado;
- prompt/prefix tuning;
- adapters compartidos;
- FourierFT en régimen pequeño.

La frase:

```text
Where LoRA cannot exist
```

solo es cierta bajo una definición estrecha:

> LoRA aplicado a todos los target modules con rank mínimo 1.

Pero LoRA sí puede existir con menos parámetros si se reducen targets, capas o se comparte estructura. Y otros PEFT compactos también existen.

---

# 4. El documento repite el problema del tamaño del test set

El documento dice:

```text
WikiText-2 Test Set (280.000 tokens)
```

pero el código de EXP-04 usa:

```python
max_test_samples=100
block_size=256
```

Eso es aproximadamente:

```text
25.600 tokens
```

no 280.000.

Este punto hay que corregir en todos los documentos v3/v4/v11/v12 si usan el mismo arnés.

La frase correcta sería:

> Evaluado sobre los primeros 100 bloques no solapados de WikiText-2 test, aproximadamente 25.6k tokens.

O bien ejecutar de verdad el test completo.

---

# 5. EXP-04 explica la inconsistencia de EXP-11/12

Aquí los targets son:

```python
targets = ["c_attn", "c_proj"]
```

Por eso salen:

```text
SpecRAMA 32x32 ≈ 294.9 KB
LoRA r=4 ≈ 1.55 MB
```

Pero en EXP-11/12 vimos:

```python
targets = ["c_attn", "c_proj", "c_fc"]
```

Eso cambia los tamaños.

Así que los documentos posteriores parecen haber heredado los tamaños de EXP-04 aunque el código posterior cambió los targets. Eso debe arreglarse.

---

# 6. Evaluación actualizada de SpecRAMA como contribución

Yo ahora lo resumiría así:

## Lo que sí está demostrado preliminarmente

SpecRAMA parece capaz de:

- mejorar GPT-2 en WikiText-2 con muy pocos parámetros;
- producir adaptadores espectrales de tamaño muy bajo;
- fusionarse en pesos para inferencia sin latencia adicional;
- funcionar con fake-quantized NF4;
- tolerar LR alto mejor que LoRA estándar en algunos protocolos.

## Lo que no está demostrado

No está demostrado que:

- supere a LoRA a igualdad de parámetros;
- sea mejor que LoRA en calidad absoluta;
- use realmente menos VRAM en 4-bit real;
- tenga un adaptador serializado tan pequeño como el tamaño de parámetros entrenables;
- haya sido evaluado sobre 280k tokens si el código usa 100 bloques;
- el benchmark v11/v12 sea limpio frente a LoRA, porque el brazo LoRA no congelaba todo.

---

# Evaluación del candidato

Esta lectura refuerza mi impresión principal:

> El candidato tiene inventiva real y capacidad de implementación, pero necesita una auditoría experimental dura antes de presentar resultados como definitivos.

Hay señales positivas:

- EXP-04 sí cuenta parámetros reales.
- Los resultados negativos frente a LoRA no fueron ocultados.
- La idea espectral tiene un nicho plausible.
- El candidato itera rápido y documenta.

Pero también señales preocupantes:

- nombres/claims más fuertes que los datos;
- “sub-KB” aunque la tabla diga 18.8 KB;
- “280k tokens” aunque el código evalúe ~25.6k;
- mezcla de targets entre experimentos;
- comparación LoRA contaminada en EXP-11/12;
- falta de seeds/split validation.

---

# Veredicto de reclutamiento final provisional

Yo sí entrevistaría al candidato.

No por los claims tal como están escritos, sino porque:

1. genera hipótesis originales;
2. implementa rápido;
3. sabe moverse en PyTorch/Transformers;
4. entiende PEFT/cuántización lo suficiente para hacer prototipos no triviales;
5. parece capaz de corregir errores si se le señalan.

Pero no aceptaría los resultados publicados como evidencia frontier-grade todavía.

Mi decisión:

```text
Avanzar a entrevista técnica: sí.
Oferta directa: no.
Prueba de reproducción limpia: obligatoria.
```

Categoría:

> **High-upside research engineer / inventor, con necesidad clara de disciplina experimental.**

---

# Qué le pediría ahora al candidato

No pediría más archivos por ahora. Pediría una **re-ejecución limpia** de SpecRAMA con un protocolo corregido:

1. Congelar todo el modelo en todos los brazos.
2. Usar los mismos targets en todos los experimentos.
3. Reportar conteo real de parámetros entrenables.
4. Reportar tamaño de:
   - parámetros entrenables;
   - buffers/permutaciones;
   - `state_dict` real del adaptador.
5. Evaluar sobre test completo o declarar claramente subset.
6. Usar validation para elegir LR.
7. Ejecutar 3 semillas.
8. Comparar contra:
   - LoRA r=1, r=2, r=4;
   - IA³;
   - VeRA;
   - BitFit;
   - SpecRAMA 8x8, 16x16, 32x32.
9. Reportar:
   - PPL;
   - bpt;
   - tokens/s;
   - peak VRAM;
   - wall-clock;
   - número de parámetros activos y serializados.

Si después de eso SpecRAMA sigue mostrando una ventaja clara en el régimen `<20 KB` o `<100 KB`, entonces hay una contribución publicable como PEFT ultra-compacto espectral.

---

# Mi evaluación numérica final provisional

| Dimensión | Nota |
|---|---:|
| Originalidad | 8.5/10 |
| Capacidad de prototipado | 8/10 |
| Conocimiento práctico de ML systems | 7/10 |
| Rigor experimental actual | 5/10 |
| Honestidad al documentar resultados mixtos | 7.5/10 |
| Claridad de visión investigadora | 8/10 |
| Preparación para paper top-tier hoy | 4.5/10 |
| Potencial con mentoría/estructura | 8/10 |

Resumen final:

> **Candidato interesante y entrevistable. SpecRAMA no está aún demostrado como superior a LoRA, pero sí contiene una idea original con señales prometedoras en PEFT ultra-compacto. El mayor gap del candidato no es creatividad ni implementación, sino control experimental y precisión de claims.**