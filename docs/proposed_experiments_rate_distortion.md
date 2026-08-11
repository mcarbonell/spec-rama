# Próximos experimentos propuestos — Rate–distortion, cuantización y SpecRAMA

> **Propósito:** convertir la intuición «la estructura espectral permite gastar bits donde importan» en comparaciones falsables. Cada resultado debe informar PPL/loss, bits reales almacenados, tamaño de adaptador, latencia antes y después de `merge()`, semillas y controles equivalentes.

## Contexto

SpecRAMA ya aporta tres piezas relevantes: reordenamiento 2D de pesos para concentrar energía, cores DCT/FWHT/wavelet de tamaño limitado y cuantización jerárquica de core/resto. Los resultados actuales muestran un trade-off práctico: en GPT-2 Small NF4/WikiText-2, SpecRAMA Wavelet alcanza 37.98 PPL con adaptador de 294.9 KB, mientras LoRA ajustado alcanza 35.58 PPL con 1.55 MB.

La pregunta siguiente no es si una representación espectral es universalmente superior a una densa. Es: **qué presupuesto de coeficientes y bits ofrece la menor distorsión para un presupuesto total concreto, y qué parte de esa ventaja procede de la base, de la permutación o de la adaptación?**

---

## EXP-13 — Curva core-size / rate–distortion de SpecRAMA

### Pregunta

¿Cómo cambia la recuperación de PPL al aumentar el tamaño de core espectral, y dónde está el punto de rendimiento decreciente por bit adicional?

### Diseño

- Base fija: GPT-2 Small cuantizado NF4 por bloques, mismo split WikiText-2 y mismo presupuesto de pasos para todos los brazos.
- Transformada inicial: Wavelet; ejecutar después DCT-2D como réplica.
- Tamaños de core: `4×4`, `8×8`, `16×16`, `32×32`, `64×64` cuando la forma de la capa lo permita.
- Mantener constantes inicialización, targets, scheduler, `alpha_m`, `alpha_a`, semilla y LR previamente validado; después repetir al menos con tres semillas.
- Informar tanto parámetros entrenables como bits reales del core al exportarlo a 4/8/16 bits.

### Qué responde

Mide la curva continua entre adaptación ultracompacta y adaptación más expresiva. Permite responder si el core `32×32` es una elección eficiente o simplemente un punto arbitrario de la configuración histórica.

### Criterio de lectura

Construir PPL contra bits totales. Un core mayor sólo se considera útil si su mejora excede `2×SE` y compensa el aumento de bits. No comparar únicamente PPL final.

### Amenaza principal

El número de pasos o LR puede favorecer cores pequeños/grandes. Añadir un mini-barrido de LR para al menos el core mínimo, medio y máximo.

---

## EXP-14 — Cuantización global frente a cuantización por bandas y por bloques

### Pregunta

¿La degradación de cores grandes a pocos bits se debe a una escala global compartida, y cuánto recuperan escalas por banda/bloque?

### Diseño

Sobre los mismos adapters entrenados de EXP-13, comparar el export/import sin reentrenar:

1. Cuantización uniforme global por matriz.
2. Esquema actual core/resto (`core_bits`, `rest_bits`).
3. Escala independiente por subbanda wavelet o anillo de frecuencia DCT.
4. Escala por bloques de tamaño fijo dentro del plano espectral.
5. Opcional: cuantización no uniforme NF4 en cada banda.

Medir error de reconstrucción del update, PPL tras `merge()`, bits de códigos, bits de escalas y cualquier metadato adicional.

### Qué responde

Separa “más coeficientes empeoran 4-bit” de “una única escala es una mala codificación”. La hipótesis es que los coeficientes de baja y alta energía requieren rangos distintos; por tanto, bandas/escalas independientes deberían desplazar la curva rate–distortion favorablemente.

### Criterio de lectura

Comparar a igualdad de **bits totales almacenados**, no sólo a igualdad de número de bits por coeficiente. Las escalas, índices y padding cuentan.

### Amenaza principal

Una mejora de MSE de pesos no garantiza recuperación de PPL. El criterio final es PPL de modelo fusionado sobre test retenido.

---

## EXP-15 — Top-K espectral, thresholding y codificación de soporte

### Pregunta

¿Es mejor almacenar sólo los coeficientes espectrales significativos que cuantizar todo el plano a pocos bits?

### Diseño

- Partir de adapters de EXP-13 y generar variantes top-K con `K={1%, 2%, 5%, 10%, 25%, 50%, 100%}` del plano espectral.
- Comparar tres soportes: top-K por magnitud, rectángulo low-frequency y top-K estructurado por subbandas.
- Codificar explícitamente valores, índices/bitmap de soporte y escalas; usar una versión simple de packing y, opcionalmente, entropy coding.
- Evaluar tras reconstrucción y `merge()`; medir PPL, MSE del update, bytes reales de checkpoint y tiempo de descompresión.

### Qué responde

Prueba directamente si la energía concentrada observada tras las permutaciones se convierte en compresión efectiva, no sólo en una visualización bonita de espectro.

### Criterio de lectura

Para cada objetivo de PPL, informar el menor número de bytes alcanzado. Para cada objetivo de bytes, informar la mejor PPL. La curva, no un único K, es el resultado.

### Amenaza principal

Top-K por magnitud puede sobreajustar al checkpoint concreto. Validar el patrón en múltiples capas, semillas y checkpoints de entrenamiento.

---

## EXP-16 — Ablación causal: permutación frente a base espectral

### Pregunta

¿La concentración de energía y la recuperación de PPL proceden de la permutación TSP, de la base wavelet/DCT o de ambos?

### Diseño factorial 2×3

| Orden de canales | Base del core |
| :--- | :--- |
| Identidad | Wavelet, DCT-2D, base ortogonal aleatoria fija |
| Bipartite TSP | Wavelet, DCT-2D, base ortogonal aleatoria fija |

Usar mismo core-size, bits, targets, semillas y presupuesto de adaptación. Registrar TV antes/después de la permutación, energía acumulada de las primeras subbandas y PPL final.

### Qué responde

Evita atribuir a DCT/wavelets un beneficio que pudiera proceder enteramente de reordenar canales, o atribuir a TSP un resultado compatible con cualquier base compacta.

### Criterio de lectura

La afirmación de una base concreta exige superar tanto la base aleatoria como la condición sin permutación con `|Δ|≥2×SE`.

### Amenaza principal

La permutación se calcula sobre pesos preentrenados y puede contener información de una capa concreta. Conservar y versionar sus hashes para reproducibilidad.

---

## EXP-17 — Coste de inferencia y formato de despliegue

### Pregunta

¿El ahorro de almacenamiento o parámetros se traduce en latencia, memoria pico o tiempo de cold-start tras fusionar el adapter?

### Diseño

Medir sobre hardware fijo y tras calentamiento:

1. Base NF4 sin adapter.
2. SpecRAMA online sin `merge()`.
3. SpecRAMA reconstruido y fusionado con `merge()`.
4. LoRA de calidad comparable y tamaño comparable cuando exista.

Separar carga de checkpoint, descompresión, construcción de permutaciones, primera inferencia y throughput sostenido. Repetir suficientes veces y reportar mediana, percentiles y memoria pico.

### Qué responde

Aclara qué beneficio llega al despliegue. Un adapter compacto puede ahorrar transferencia y almacenamiento aunque el forward online no sea más rápido; `merge()` puede eliminar latencia de adapter pero no comprime por sí solo la matriz fusionada.

### Amenaza principal

No confundir FLOPs teóricos con wall-clock. Si DCT/DWT se materializa como matrices PyTorch, el benchmark mide esa implementación, no un kernel fusionado ideal.

---

## Orden recomendado

1. **EXP-14**: barato, usa adapters ya entrenados y aborda directamente la señal de cuantización global.
2. **EXP-15**: transforma concentración de energía en bytes reales.
3. **EXP-16**: establece atribución causal de base y permutación.
4. **EXP-13**: curva completa con reentrenamiento; usar los hallazgos de 14–16 para elegir bits/base.
5. **EXP-17**: medir despliegue una vez fijado un formato de adapter competitivo.

## Requisitos comunes de registro

Cada experimento debe guardar configuración completa, commit, versión de bases/permutaciones, semillas, split de datos, PPL/loss por época, bytes exactos por componente, tiempos separados y JSON crudo. Las conclusiones deben incluir controles, SE entre semillas y amenazas a la validez; un resultado de una única semilla sólo es [SEÑAL] o [RUIDO-SOSPECHA].
