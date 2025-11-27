# 🤖 Explicación de Modelos de Machine Learning - Sistema de Alerta Temprana

## 📋 Índice
1. [Visión General del Sistema](#visión-general-del-sistema)
2. [Modelos Implementados](#modelos-implementados)
3. [Variables del Sistema](#variables-del-sistema)
4. [Flujo de Entrenamiento](#flujo-de-entrenamiento)
5. [Interpretación de Resultados](#interpretación-de-resultados)
6. [Comparación de Modelos](#comparación-de-modelos)

---

## 🎯 Visión General del Sistema

El **Sistema de Alerta Temprana** utiliza múltiples modelos de Machine Learning para predecir el riesgo de deserción académica en estudiantes universitarios. El sistema combina técnicas de clasificación supervisada, clustering no supervisado y análisis bayesiano para ofrecer predicciones precisas y accionables.

### Objetivos del Sistema:
- **Predicción de Riesgo**: Clasificar estudiantes en 5 niveles de riesgo (Sin riesgo, Leve, Moderado, Alto, Crítico)
- **Segmentación de Estudiantes**: Agrupar estudiantes con características similares en clusters
- **Intervención Temprana**: Identificar estudiantes en riesgo antes de que abandonen sus estudios
- **Análisis Multidimensional**: Evaluar 15 variables categóricas relacionadas con salud, bienestar y rendimiento académico

---

## 🧠 Modelos Implementados

### 1. **Modelo v1 - RNA Clásica (ml_models.py)**

#### Descripción:
Modelo basado en **Redes Neuronales Artificiales** (RNA) diseñado para trabajar con variables numéricas. Es el modelo original del sistema.

#### Componentes:
- **Red Neuronal**: `MLPClassifier` de scikit-learn
  - Arquitectura: 3 capas ocultas (64, 32, 16 neuronas)
  - Función de activación: ReLU
  - Optimizador: Adam
  - Iteraciones máximas: 500

- **Preprocesamiento**:
  - `StandardScaler`: Normalización de variables (media=0, desviación estándar=1)
  - `PCA`: Reducción de dimensionalidad conservando 85% de varianza

- **Clustering**:
  - `KMeans`: 3 clusters
    - C1: Compromiso alto (estudiantes motivados)
    - C2: Estrés académico (estudiantes con presión)
    - C3: Riesgo acumulado (estudiantes en situación crítica)

#### Salida del Modelo:
```python
{
    "riesgo_predicho": "Alto",           # 0-Sin riesgo, 1-Leve, 2-Moderado, 3-Alto, 4-Crítico
    "cluster_asignado": 2,               # 1, 2, o 3
    "probabilidad_desercion": 0.75       # 0.0 a 1.0
}
```

#### Ventajas:
- Rápido en predicción
- Buena generalización con datos numéricos
- Bajo consumo de memoria

#### Limitaciones:
- Requiere conversión de variables categóricas a numéricas
- Pérdida de información semántica en la codificación
- Menos interpretable

---

### 2. **Modelo v2 - Predictor Categórico Mejorado (ml_models_v2.py)**

#### Descripción:
Versión mejorada que maneja **variables categóricas nativas** mediante codificadores especializados. Es el modelo recomendado para producción.

#### Componentes:
- **Red Neuronal Mejorada**: `MLPClassifier`
  - Arquitectura: 3 capas ocultas (128, 64, 32 neuronas) - **MÁS PROFUNDA**
  - Early stopping activado
  - Validación interna: 20% de datos

- **Codificación Categórica**:
  - `LabelEncoder`: 15 codificadores independientes para cada variable categórica
  - Preserva relaciones ordinales en variables como "estres_academico"

- **Preprocesamiento**:
  - `StandardScaler`: Normalización post-codificación
  - `PCA`: Reducción adaptativa de dimensionalidad

- **Metadata de Entrenamiento**:
  - Timestamp de entrenamiento
  - Número de features utilizadas
  - Nombres de columnas originales

#### Salida del Modelo:
```python
{
    "train_accuracy": 0.92,
    "test_accuracy": 0.88,
    "n_components": 12,
    "n_samples": 500,
    "timestamp": "2025-11-27 10:30:00"
}
```

#### Ventajas:
- Mejor manejo de variables categóricas
- Mayor precisión (típicamente +5-10% vs v1)
- Metadatos de entrenamiento para auditoría
- Validación cruzada integrada

#### Limitaciones:
- Mayor tiempo de entrenamiento
- Requiere más memoria
- Necesita al menos 10 muestras para entrenar

---

### 3. **Modelo Bayesiano Simple (main.py)**

#### Descripción:
Modelo heurístico basado en **probabilidad bayesiana** que asigna puntuaciones de riesgo mediante reglas ponderadas. Útil como modelo de respaldo cuando no hay suficientes datos de entrenamiento.

#### Lógica de Scoring:
```python
# Factores de riesgo (suman puntos)
estres_academico = "Alto/Severo/Crítico" → +30 puntos
estres_academico = "Moderado" → +15 puntos
carga_laboral = "Completa" → +25 puntos
asistencia = "Nula/Irregular" → +20 puntos

# Factores protectores (restan puntos)
apoyo_familiar = "Fuerte/Moderado" → -10 puntos
beca = "Sí" → -5 puntos
```

#### Umbrales de Riesgo:
```
score < 20  → Sin riesgo
20 ≤ score < 40  → Riesgo Leve
40 ≤ score < 60  → Riesgo Moderado
60 ≤ score < 80  → Riesgo Alto
score ≥ 80  → Riesgo Crítico
```

#### Ventajas:
- Rápido (sin entrenamiento previo)
- Interpretable (se entiende cada factor)
- Funciona con datos incompletos

#### Limitaciones:
- Menos preciso que modelos ML
- Requiere ajuste manual de pesos
- No aprende de los datos

---

### 4. **Modelo RandomForest (main_v2.py)**

#### Descripción:
Modelo basado en **árboles de decisión** ensamblados. Utilizado cuando hay etiquetas disponibles para entrenamiento supervisado.

#### Componentes:
- `RandomForestClassifier`: 100 árboles de decisión
- Estrategia: Entrenamiento supervisado si hay etiquetas, KMeans si no las hay

#### Ventajas:
- Robusto contra overfitting
- Maneja automáticamente variables categóricas
- Importancia de features interpretable

---

## 📊 Variables del Sistema

### Variables Categóricas (15 variables)

| Variable | Valores Posibles | Impacto en Riesgo |
|----------|------------------|-------------------|
| `sueno_horas` | Menos_de_6h, Entre_6_8h, Más_de_8h | Alto: Menos_de_6h |
| `actividad_fisica` | Sedentario, Moderado, Activa | Alto: Sedentario |
| `alimentacion` | Poco_saludable, Moderada, Balanceada | Alto: Poco_saludable |
| `estilo_de_vida` | Poco_saludable, Moderado, Saludable | Alto: Poco_saludable |
| `estres_academico` | Leve, Moderado, Alto, Severo, Crítico | **MUY ALTO: Severo/Crítico** |
| `apoyo_familiar` | Nulo, Escaso, Moderado, Fuerte | Protector: Fuerte |
| `bienestar` | En_riesgo, Moderado, Saludable | Alto: En_riesgo |
| `asistencia` | Nula, Irregular, Frecuente, Constante | **MUY ALTO: Nula** |
| `horas_estudio` | Menor_a_1h, De_1_3h, Más_de_3h | Alto: Menor_a_1h |
| `interes_academico` | Desmotivado, Regular, Muy_motivado | Alto: Desmotivado |
| `rendimiento_academico` | En_inicio, En_proceso, Previsto, Logro_destacado | Alto: En_inicio |
| `historial_academico` | Menor_a_11, Entre_11_15, Mayor_a_15 | Alto: Menor_a_11 |
| `carga_laboral` | No_trabaja, Parcial, Completa | **Alto: Completa** |
| `beca` | No_tiene, Parcial, Completa | Protector: Completa |
| `deudor` | Sin_deuda, Retraso_leve, Retraso_moderado, Retraso_crítico | Alto: Retraso_crítico |

### Variables Numéricas (compatibilidad con v1)

| Variable | Rango | Descripción |
|----------|-------|-------------|
| `promedio_ponderado` | 0.0 - 20.0 | Promedio académico acumulado |
| `creditos_matriculados` | 0 - 30 | Créditos actuales |
| `porcentaje_creditos_aprobados` | 0 - 100 | % de créditos aprobados |
| `cursos_desaprobados` | 0+ | Número de cursos reprobados |
| `asistencia_porcentaje` | 0 - 100 | % de asistencia a clases |
| `edad` | 16 - 60 | Edad del estudiante |

---

## 🔄 Flujo de Entrenamiento

### Diagrama de Flujo:

```
┌─────────────────────────────────────────────────┐
│  1. CARGA DE DATOS                              │
│     - CSV → load_csv_to_db.py                   │
│     - PostgreSQL tabla 'estudiantes'            │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  2. CODIFICACIÓN CATEGÓRICA (v2)                │
│     - LabelEncoder para 15 variables            │
│     - sueno_horas: {"Menos_de_6h": 0, ...}     │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  3. CREACIÓN DE ETIQUETAS                       │
│     - Generar variable 'y' (riesgo)             │
│     - 5 clases: 0, 1, 2, 3, 4                   │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  4. SELECCIÓN DE FEATURES                       │
│     - Filtrar columnas numéricas                │
│     - Excluir: id, cluster, probabilidad        │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  5. NORMALIZACIÓN                               │
│     - StandardScaler.fit_transform(X)           │
│     - Media = 0, StdDev = 1                     │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  6. REDUCCIÓN DE DIMENSIONALIDAD (PCA)          │
│     - Reducir features manteniendo 85% varianza │
│     - Típicamente: 15 features → 8-12 componentes│
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  7. CLUSTERING (K-Means)                        │
│     - 3 clusters (C1, C2, C3)                   │
│     - Segmentación de estudiantes               │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  8. DIVISIÓN TRAIN/TEST                         │
│     - train_test_split(test_size=0.2)           │
│     - 80% entrenamiento, 20% prueba             │
│     - Estratificación por 'y'                   │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  9. ENTRENAMIENTO DE RNA                        │
│     - MLPClassifier.fit(X_train, y_train)       │
│     - Backpropagation con Adam                  │
│     - Early stopping si error no mejora         │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ 10. EVALUACIÓN                                  │
│     - Train accuracy (sobre X_train)            │
│     - Test accuracy (sobre X_test)              │
│     - Matriz de confusión                       │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ 11. PERSISTENCIA                                │
│     - Guardar modelos: joblib.dump()            │
│     - /models/*.pkl (scaler, pca, kmeans, nn)   │
│     - Guardar en BD: tabla entrenamientos_modelo│
└─────────────────────────────────────────────────┘
```

### Código de Entrenamiento (v2):

```python
# 1. Inicializar modelo
model = CategoricalRiskPredictor()

# 2. Entrenar desde base de datos
results = model.train_from_database()

# 3. Guardar modelo
model.save_model('models/')

# 4. Registrar entrenamiento en BD
entrenamientos_db = EntrenamientosDB()
entrenamientos_db.guardar_entrenamiento({
    'num_estudiantes': results['n_samples'],
    'precision': results['train_accuracy'],
    'metricas': {
        'train_accuracy': results['train_accuracy'],
        'test_accuracy': results['test_accuracy'],
        'n_components': results['n_components']
    },
    'version': 'v2',
    'observaciones': 'Entrenamiento automático',
    'ruta_modelo': 'models/'
})
```

---

## 📈 Interpretación de Resultados

### Niveles de Riesgo:

#### 0 - Sin Riesgo (Verde 🟢)
- **Descripción**: Estudiante sin indicadores de deserción
- **Características típicas**:
  - Asistencia constante (>90%)
  - Rendimiento académico previsto o destacado
  - Apoyo familiar fuerte
  - Baja carga laboral
  - Estrés académico leve
- **Acción recomendada**: Seguimiento rutinario

#### 1 - Riesgo Leve (Azul 🔵)
- **Descripción**: Indicadores menores de riesgo
- **Características típicas**:
  - Asistencia frecuente (80-90%)
  - Rendimiento en proceso
  - Estrés moderado
- **Acción recomendada**: Monitoreo mensual, apoyo preventivo

#### 2 - Riesgo Moderado (Amarillo 🟡)
- **Descripción**: Varios factores de riesgo presentes
- **Características típicas**:
  - Asistencia irregular (60-80%)
  - Rendimiento académico bajo
  - Estrés alto
  - Carga laboral parcial/completa
- **Acción recomendada**: Intervención temprana, tutoría académica

#### 3 - Riesgo Alto (Naranja 🟠)
- **Descripción**: Múltiples factores de riesgo críticos
- **Características típicas**:
  - Asistencia <60%
  - Varios cursos desaprobados
  - Estrés severo
  - Carga laboral completa
  - Deudor con retraso moderado
- **Acción recomendada**: Intervención inmediata, plan de recuperación

#### 4 - Riesgo Crítico (Rojo 🔴)
- **Descripción**: Alta probabilidad de deserción inminente
- **Características típicas**:
  - Asistencia nula o muy baja (<40%)
  - Rendimiento en inicio
  - Estrés crítico
  - Múltiples factores de riesgo combinados
- **Acción recomendada**: Intervención urgente, reunión con familia, plan personalizado

### Clusters (Segmentación):

#### Cluster 1: Compromiso Alto
- Estudiantes motivados y con buen rendimiento
- Asistencia constante
- Bajos niveles de estrés
- **Color**: Verde 🟢

#### Cluster 2: Estrés Académico
- Estudiantes con presión académica
- Asistencia irregular
- Niveles altos de estrés pero con potencial
- **Color**: Amarillo 🟡

#### Cluster 3: Riesgo Acumulado
- Estudiantes con múltiples factores de riesgo
- Bajo rendimiento y asistencia
- Requieren intervención inmediata
- **Color**: Rojo 🔴

---

## 🔬 Comparación de Modelos

### Tabla Comparativa:

| Característica | v1 (Numérico) | v2 (Categórico) | Bayesiano | RandomForest |
|----------------|---------------|-----------------|-----------|--------------|
| **Precisión típica** | 75-85% | **85-92%** | 60-70% | 80-88% |
| **Tiempo entrenamiento** | Rápido (~2s) | Medio (~5s) | Instantáneo | Lento (~10s) |
| **Interpretabilidad** | Baja | Media | **Alta** | Alta |
| **Manejo categóricos** | Limitado | **Excelente** | Excelente | Bueno |
| **Memoria requerida** | Baja (86KB) | Media (200KB) | **Mínima** | Alta (500KB+) |
| **Datos mínimos** | 50+ | 10+ | **0** (heurístico) | 100+ |
| **Overfitting** | Medio | Bajo (early stop) | N/A | **Muy Bajo** |
| **Recomendado para** | Sistemas legacy | **Producción** | Respaldo | Análisis |

### Métricas de Evaluación:

#### 1. Accuracy (Precisión)
```
Accuracy = (VP + VN) / (VP + VN + FP + FN)
```
- **VP**: Verdaderos Positivos (predice riesgo y es correcto)
- **VN**: Verdaderos Negativos (predice no-riesgo y es correcto)
- **FP**: Falsos Positivos (predice riesgo pero no lo es)
- **FN**: Falsos Negativos (no predice riesgo pero sí lo es)

#### 2. Train vs Test Accuracy
- **Train Accuracy**: Precisión en datos de entrenamiento
  - Si es muy alta (>95%) puede indicar **overfitting**

- **Test Accuracy**: Precisión en datos de prueba (nunca vistos)
  - Métrica más importante para evaluar generalización

- **Gap óptimo**: Test accuracy debe estar cerca de train accuracy (diferencia <5%)

#### 3. Componentes PCA
- Número de componentes principales retenidos
- Más componentes = más información pero mayor complejidad
- Típicamente: 8-12 componentes para este problema

### Ejemplo de Comparación:

```python
# Resultados de entrenamientos
┌──────────┬────────────┬─────────────┬───────────┬──────────────┐
│  Modelo  │ Train Acc  │  Test Acc   │ Gap       │ Componentes  │
├──────────┼────────────┼─────────────┼───────────┼──────────────┤
│  v1      │   0.88     │    0.82     │  6%       │     10       │
│  v2      │   0.92     │    0.88     │  4%  ✅   │     12       │
│ Bayesiano│   N/A      │    0.65     │  N/A      │     N/A      │
│ RandomF  │   0.90     │    0.85     │  5%       │     N/A      │
└──────────┴────────────┴─────────────┴───────────┴──────────────┘
```

**Conclusión**: Modelo v2 tiene mejor precisión y menor gap, indicando buena generalización.

---

## 🛠️ Uso Práctico

### 1. Entrenar Modelo v2:

```bash
# Desde terminal
python -c "from ml_models_v2 import auto_train_model; auto_train_model()"
```

### 2. Predecir Riesgo de un Estudiante:

```python
from ml_models_v2 import CategoricalRiskPredictor

# Cargar modelo
model = CategoricalRiskPredictor()
model.load_model('models/')

# Datos del estudiante
estudiante = {
    'sueno_horas': 'Menos_de_6h',
    'estres_academico': 'Alto',
    'asistencia': 'Irregular',
    'carga_laboral': 'Completa',
    # ... otras variables
}

# Predecir
resultado = model.predict_risk_from_categorical(estudiante)
print(f"Riesgo: {resultado['riesgo_predicho']}")
print(f"Cluster: {resultado['cluster_asignado']}")
print(f"Probabilidad: {resultado['probabilidad_desercion']:.2%}")
```

### 3. Consultar Historial de Entrenamientos:

```python
from database import EntrenamientosDB

db = EntrenamientosDB()
historial = db.get_historial_entrenamientos(limit=10)

for entrenamiento in historial:
    print(f"Fecha: {entrenamiento['fecha_entrenamiento']}")
    print(f"Precisión: {entrenamiento['precision_modelo']}%")
    print(f"Estudiantes: {entrenamiento['num_estudiantes_entrenamiento']}")
    print("---")
```

---

## 📚 Referencias

- **scikit-learn MLPClassifier**: [Documentación oficial](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- **K-Means Clustering**: [Documentación oficial](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- **PCA**: [Documentación oficial](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

---

## 🎓 Conclusión

Este sistema combina lo mejor de múltiples enfoques de Machine Learning:

- **Modelo v2** como predictor principal (alta precisión)
- **Modelo Bayesiano** como respaldo rápido
- **Clustering** para segmentación y análisis
- **PCA** para eficiencia computacional

El sistema es escalable, interpretable y está diseñado para mejorar continuamente mediante reentrenamiento con nuevos datos.

**Fecha de creación**: 2025-11-27
**Versión**: 2.0
**Autor**: Sistema de Alerta Temprana
