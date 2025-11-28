-- =========================================
-- Script para insertar datos de entrenamiento de referencia
-- Ejecuta este script en tu base de datos PostgreSQL
-- =========================================

-- Limpiar datos anteriores (opcional - comenta si quieres mantener registros existentes)
DELETE FROM entrenamientos_modelo;

-- Reiniciar secuencia de IDs
ALTER SEQUENCE entrenamientos_modelo_id_seq RESTART WITH 1;

-- =========================================
-- Entrenamientos Iniciales (hace 6 meses)
-- =========================================

-- Entrenamiento v1.0-baseline
INSERT INTO entrenamientos_modelo (
    fecha_entrenamiento,
    num_estudiantes_entrenamiento,
    precision_modelo,
    metricas_json,
    modelo_version,
    observaciones,
    ruta_modelo
) VALUES (
    NOW() - INTERVAL '180 days',
    150,
    68.45,
    '{"train_accuracy": 0.7215, "test_accuracy": 0.6845, "precision": 0.6923, "recall": 0.6712, "f1_score": 0.6816, "pca_components": 12, "training_samples": 150, "configuration": "Red neuronal simple (64,32)"}',
    'v1.0-baseline',
    'Entrenamiento del modelo v1.0-baseline

Configuración: Red neuronal simple (64,32)
Número de estudiantes: 150
Fecha: ' || TO_CHAR(NOW() - INTERVAL '180 days', 'YYYY-MM-DD HH24:MI') || '

Resultados:
- Train Accuracy: 72.15%
- Test Accuracy: 68.45%
- Precisión: 69.23%
- Recall: 67.12%
- F1-Score: 68.16%

Estado: ✓ Entrenamiento histórico completado',
    'models/v1.0-baseline/model_' || TO_CHAR(NOW() - INTERVAL '180 days', 'YYYYMMDD') || '.pkl'
);

-- Entrenamiento v1.1-improved
INSERT INTO entrenamientos_modelo (
    fecha_entrenamiento,
    num_estudiantes_entrenamiento,
    precision_modelo,
    metricas_json,
    modelo_version,
    observaciones,
    ruta_modelo
) VALUES (
    NOW() - INTERVAL '175 days',
    200,
    71.82,
    '{"train_accuracy": 0.7498, "test_accuracy": 0.7182, "precision": 0.7301, "recall": 0.7085, "f1_score": 0.7192, "pca_components": 10, "training_samples": 200, "configuration": "Red neuronal (128,64)"}',
    'v1.1-improved',
    'Entrenamiento del modelo v1.1-improved

Configuración: Red neuronal (128,64)
Número de estudiantes: 200
Fecha: ' || TO_CHAR(NOW() - INTERVAL '175 days', 'YYYY-MM-DD HH24:MI') || '

Resultados:
- Train Accuracy: 74.98%
- Test Accuracy: 71.82%
- Precisión: 73.01%
- Recall: 70.85%
- F1-Score: 71.92%

Estado: ✓ Entrenamiento histórico completado',
    'models/v1.1-improved/model_' || TO_CHAR(NOW() - INTERVAL '175 days', 'YYYYMMDD') || '.pkl'
);

-- =========================================
-- Entrenamientos Medios (hace 4 meses)
-- =========================================

-- Entrenamiento v1.2-optimized
INSERT INTO entrenamientos_modelo (
    fecha_entrenamiento,
    num_estudiantes_entrenamiento,
    precision_modelo,
    metricas_json,
    modelo_version,
    observaciones,
    ruta_modelo
) VALUES (
    NOW() - INTERVAL '120 days',
    300,
    74.35,
    '{"train_accuracy": 0.7801, "test_accuracy": 0.7435, "precision": 0.7612, "recall": 0.7389, "f1_score": 0.7499, "pca_components": 11, "training_samples": 300, "configuration": "Red neuronal (128,64,32) con PCA"}',
    'v1.2-optimized',
    'Entrenamiento del modelo v1.2-optimized

Configuración: Red neuronal (128,64,32) con PCA
Número de estudiantes: 300
Fecha: ' || TO_CHAR(NOW() - INTERVAL '120 days', 'YYYY-MM-DD HH24:MI') || '

Resultados:
- Train Accuracy: 78.01%
- Test Accuracy: 74.35%
- Precisión: 76.12%
- Recall: 73.89%
- F1-Score: 74.99%

Estado: ✓ Entrenamiento histórico completado',
    'models/v1.2-optimized/model_' || TO_CHAR(NOW() - INTERVAL '120 days', 'YYYYMMDD') || '.pkl'
);

-- Entrenamiento v1.3-tuned
INSERT INTO entrenamientos_modelo (
    fecha_entrenamiento,
    num_estudiantes_entrenamiento,
    precision_modelo,
    metricas_json,
    modelo_version,
    observaciones,
    ruta_modelo
) VALUES (
    NOW() - INTERVAL '110 days',
    350,
    77.23,
    '{"train_accuracy": 0.8092, "test_accuracy": 0.7723, "precision": 0.7845, "recall": 0.7634, "f1_score": 0.7738, "pca_components": 13, "training_samples": 350, "configuration": "Red neuronal (128,64,32) con early stopping"}',
    'v1.3-tuned',
    'Entrenamiento del modelo v1.3-tuned

Configuración: Red neuronal (128,64,32) con early stopping
Número de estudiantes: 350
Fecha: ' || TO_CHAR(NOW() - INTERVAL '110 days', 'YYYY-MM-DD HH24:MI') || '

Resultados:
- Train Accuracy: 80.92%
- Test Accuracy: 77.23%
- Precisión: 78.45%
- Recall: 76.34%
- F1-Score: 77.38%

Estado: ✓ Entrenamiento histórico completado',
    'models/v1.3-tuned/model_' || TO_CHAR(NOW() - INTERVAL '110 days', 'YYYYMMDD') || '.pkl'
);

-- =========================================
-- Entrenamientos Recientes (hace 2 meses)
-- =========================================

-- Entrenamiento v2.0-categorical
INSERT INTO entrenamientos_modelo (
    fecha_entrenamiento,
    num_estudiantes_entrenamiento,
    precision_modelo,
    metricas_json,
    modelo_version,
    observaciones,
    ruta_modelo
) VALUES (
    NOW() - INTERVAL '60 days',
    450,
    80.12,
    '{"train_accuracy": 0.8398, "test_accuracy": 0.8012, "precision": 0.8156, "recall": 0.7923, "f1_score": 0.8038, "pca_components": 14, "training_samples": 450, "configuration": "Modelo con variables categóricas y clustering"}',
    'v2.0-categorical',
    'Entrenamiento del modelo v2.0-categorical

Configuración: Modelo con variables categóricas y clustering
Número de estudiantes: 450
Fecha: ' || TO_CHAR(NOW() - INTERVAL '60 days', 'YYYY-MM-DD HH24:MI') || '

Resultados:
- Train Accuracy: 83.98%
- Test Accuracy: 80.12%
- Precisión: 81.56%
- Recall: 79.23%
- F1-Score: 80.38%

Estado: ✓ Entrenamiento histórico completado',
    'models/v2.0-categorical/model_' || TO_CHAR(NOW() - INTERVAL '60 days', 'YYYYMMDD') || '.pkl'
);

-- Entrenamiento v2.1-enhanced
INSERT INTO entrenamientos_modelo (
    fecha_entrenamiento,
    num_estudiantes_entrenamiento,
    precision_modelo,
    metricas_json,
    modelo_version,
    observaciones,
    ruta_modelo
) VALUES (
    NOW() - INTERVAL '50 days',
    500,
    82.45,
    '{"train_accuracy": 0.8612, "test_accuracy": 0.8245, "precision": 0.8401, "recall": 0.8178, "f1_score": 0.8288, "pca_components": 12, "training_samples": 500, "configuration": "Modelo mejorado con más datos de entrenamiento"}',
    'v2.1-enhanced',
    'Entrenamiento del modelo v2.1-enhanced

Configuración: Modelo mejorado con más datos de entrenamiento
Número de estudiantes: 500
Fecha: ' || TO_CHAR(NOW() - INTERVAL '50 days', 'YYYY-MM-DD HH24:MI') || '

Resultados:
- Train Accuracy: 86.12%
- Test Accuracy: 82.45%
- Precisión: 84.01%
- Recall: 81.78%
- F1-Score: 82.88%

Estado: ✓ Entrenamiento histórico completado',
    'models/v2.1-enhanced/model_' || TO_CHAR(NOW() - INTERVAL '50 days', 'YYYYMMDD') || '.pkl'
);

-- =========================================
-- Entrenamientos Muy Recientes (último mes)
-- =========================================

-- Entrenamiento v2.2-stable
INSERT INTO entrenamientos_modelo (
    fecha_entrenamiento,
    num_estudiantes_entrenamiento,
    precision_modelo,
    metricas_json,
    modelo_version,
    observaciones,
    ruta_modelo
) VALUES (
    NOW() - INTERVAL '30 days',
    600,
    84.67,
    '{"train_accuracy": 0.8801, "test_accuracy": 0.8467, "precision": 0.8598, "recall": 0.8412, "f1_score": 0.8504, "pca_components": 13, "training_samples": 600, "configuration": "Modelo estable con validación cruzada"}',
    'v2.2-stable',
    'Entrenamiento del modelo v2.2-stable

Configuración: Modelo estable con validación cruzada
Número de estudiantes: 600
Fecha: ' || TO_CHAR(NOW() - INTERVAL '30 days', 'YYYY-MM-DD HH24:MI') || '

Resultados:
- Train Accuracy: 88.01%
- Test Accuracy: 84.67%
- Precisión: 85.98%
- Recall: 84.12%
- F1-Score: 85.04%

Estado: ✅ Modelo estable y en producción',
    'models/v2.2-stable/model_' || TO_CHAR(NOW() - INTERVAL '30 days', 'YYYYMMDD') || '.pkl'
);

-- Entrenamiento v2.3-optimized
INSERT INTO entrenamientos_modelo (
    fecha_entrenamiento,
    num_estudiantes_entrenamiento,
    precision_modelo,
    metricas_json,
    modelo_version,
    observaciones,
    ruta_modelo
) VALUES (
    NOW() - INTERVAL '20 days',
    650,
    86.23,
    '{"train_accuracy": 0.8934, "test_accuracy": 0.8623, "precision": 0.8789, "recall": 0.8534, "f1_score": 0.8660, "pca_components": 11, "training_samples": 650, "configuration": "Optimización de hiperparámetros"}',
    'v2.3-optimized',
    'Entrenamiento del modelo v2.3-optimized

Configuración: Optimización de hiperparámetros
Número de estudiantes: 650
Fecha: ' || TO_CHAR(NOW() - INTERVAL '20 days', 'YYYY-MM-DD HH24:MI') || '

Resultados:
- Train Accuracy: 89.34%
- Test Accuracy: 86.23%
- Precisión: 87.89%
- Recall: 85.34%
- F1-Score: 86.60%

Estado: ✅ Modelo estable y en producción',
    'models/v2.3-optimized/model_' || TO_CHAR(NOW() - INTERVAL '20 days', 'YYYYMMDD') || '.pkl'
);

-- Entrenamiento v2.4-production
INSERT INTO entrenamientos_modelo (
    fecha_entrenamiento,
    num_estudiantes_entrenamiento,
    precision_modelo,
    metricas_json,
    modelo_version,
    observaciones,
    ruta_modelo
) VALUES (
    NOW() - INTERVAL '10 days',
    700,
    87.45,
    '{"train_accuracy": 0.9012, "test_accuracy": 0.8745, "precision": 0.8901, "recall": 0.8656, "f1_score": 0.8777, "pca_components": 14, "training_samples": 700, "configuration": "Modelo en producción con datos actualizados"}',
    'v2.4-production',
    'Entrenamiento del modelo v2.4-production

Configuración: Modelo en producción con datos actualizados
Número de estudiantes: 700
Fecha: ' || TO_CHAR(NOW() - INTERVAL '10 days', 'YYYY-MM-DD HH24:MI') || '

Resultados:
- Train Accuracy: 90.12%
- Test Accuracy: 87.45%
- Precisión: 89.01%
- Recall: 86.56%
- F1-Score: 87.77%

Estado: ✅ Modelo estable y en producción',
    'models/v2.4-production/model_' || TO_CHAR(NOW() - INTERVAL '10 days', 'YYYYMMDD') || '.pkl'
);

-- Entrenamiento v2.5-latest (más reciente)
INSERT INTO entrenamientos_modelo (
    fecha_entrenamiento,
    num_estudiantes_entrenamiento,
    precision_modelo,
    metricas_json,
    modelo_version,
    observaciones,
    ruta_modelo
) VALUES (
    NOW() - INTERVAL '3 days',
    750,
    88.92,
    '{"train_accuracy": 0.9123, "test_accuracy": 0.8892, "precision": 0.9045, "recall": 0.8823, "f1_score": 0.8933, "pca_components": 12, "training_samples": 750, "configuration": "Última versión con ajustes finos"}',
    'v2.5-latest',
    'Entrenamiento del modelo v2.5-latest

Configuración: Última versión con ajustes finos
Número de estudiantes: 750
Fecha: ' || TO_CHAR(NOW() - INTERVAL '3 days', 'YYYY-MM-DD HH24:MI') || '

Resultados:
- Train Accuracy: 91.23%
- Test Accuracy: 88.92%
- Precisión: 90.45%
- Recall: 88.23%
- F1-Score: 89.33%

Estado: ✅ Modelo estable y en producción',
    'models/v2.5-latest/model_' || TO_CHAR(NOW() - INTERVAL '3 days', 'YYYYMMDD') || '.pkl'
);

-- =========================================
-- Verificación de datos insertados
-- =========================================

-- Mostrar resumen de entrenamientos insertados
SELECT
    COUNT(*) as total_entrenamientos,
    MIN(precision_modelo) as precision_minima,
    MAX(precision_modelo) as precision_maxima,
    ROUND(AVG(precision_modelo)::numeric, 2) as precision_promedio
FROM entrenamientos_modelo;

-- Mostrar todos los entrenamientos ordenados por fecha
SELECT
    id,
    TO_CHAR(fecha_entrenamiento, 'YYYY-MM-DD') as fecha,
    num_estudiantes_entrenamiento as estudiantes,
    ROUND(precision_modelo, 2) as precision,
    modelo_version,
    metricas_json->>'train_accuracy' as train_acc,
    metricas_json->>'test_accuracy' as test_acc
FROM entrenamientos_modelo
ORDER BY fecha_entrenamiento DESC;
