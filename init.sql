-- =========================================
-- Base de datos para Sistema de Alerta Temprana
-- Tabla única con todas las variables del estudiante
-- =========================================

-- Crear extensión para UUID si es necesario
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Tabla principal de estudiantes
CREATE TABLE IF NOT EXISTS estudiantes (
    id SERIAL PRIMARY KEY,
    codigo VARCHAR(50) UNIQUE NOT NULL,
    nombre VARCHAR(200) NOT NULL,
    carrera VARCHAR(100),
    ciclo INTEGER,

    -- Variables categóricas para el modelo (15 variables)
    sueno_horas VARCHAR(50),  -- Menos_de_6h, Entre_6_8h, Más_de_8h
    actividad_fisica VARCHAR(50),  -- Sedentario, Moderado, Activa
    alimentacion VARCHAR(50),  -- Poco_saludable, Moderada, Balanceada
    estilo_de_vida VARCHAR(50),  -- Poco_saludable, Moderado, Saludable
    estres_academico VARCHAR(50),  -- Leve, Moderado, Alto, Severo, Crítico
    apoyo_familiar VARCHAR(50),  -- Nulo, Escaso, Moderado, Fuerte
    bienestar VARCHAR(50),  -- En_riesgo, Moderado, Saludable
    asistencia VARCHAR(50),  -- Nula, Irregular, Frecuente, Constante
    horas_estudio VARCHAR(50),  -- Menor_a_1h, De_1_3h, Más_de_3h
    interes_academico VARCHAR(50),  -- Desmotivado, Regular, Muy_motivado
    rendimiento_academico VARCHAR(50),  -- En_inicio, En_proceso, Previsto, Logro_destacado
    historial_academico VARCHAR(50),  -- Menor_a_11, Entre_11_15, Mayor_a_15
    carga_laboral VARCHAR(50),  -- No_trabaja, Parcial, Completa
    beca VARCHAR(50),  -- No_tiene, Parcial, Completa
    deudor VARCHAR(50),  -- Sin_deuda, Retraso_leve, Retraso_moderado, Retraso_crítico

    -- Variables numéricas adicionales (para compatibilidad con modelo antiguo)
    promedio_ponderado DECIMAL(4,2),
    creditos_matriculados INTEGER,
    porcentaje_creditos_aprobados DECIMAL(5,2),
    cursos_desaprobados INTEGER,
    asistencia_porcentaje DECIMAL(5,2),
    retiros_cursos INTEGER,
    edad INTEGER,
    horas_trabajo_semana DECIMAL(5,2),
    anio_ingreso INTEGER,
    numero_ciclos_academicos INTEGER,
    cursos_matriculados_ciclo INTEGER,
    horas_estudio_semana DECIMAL(5,2),
    indice_regularidad DECIMAL(5,2),
    intentos_aprobacion_curso INTEGER,
    nota_promedio DECIMAL(4,2),

    -- Resultados del modelo
    riesgo_predicho VARCHAR(50),
    cluster_asignado INTEGER,
    probabilidad_desercion DECIMAL(5,2),

    -- Metadatos
    notas_tutor TEXT,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ultima_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    estado_seguimiento VARCHAR(50) DEFAULT 'Pendiente'
);

-- Índices para mejorar rendimiento
CREATE INDEX IF NOT EXISTS idx_codigo ON estudiantes(codigo);
CREATE INDEX IF NOT EXISTS idx_riesgo_predicho ON estudiantes(riesgo_predicho);
CREATE INDEX IF NOT EXISTS idx_cluster_asignado ON estudiantes(cluster_asignado);
CREATE INDEX IF NOT EXISTS idx_carrera ON estudiantes(carrera);
CREATE INDEX IF NOT EXISTS idx_estado_seguimiento ON estudiantes(estado_seguimiento);

-- Trigger para actualizar ultima_actualizacion
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.ultima_actualizacion = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_timestamp
BEFORE UPDATE ON estudiantes
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

-- Datos de ejemplo para testing
INSERT INTO estudiantes (
    codigo, nombre, carrera, ciclo,
    sueno_horas, actividad_fisica, alimentacion, estilo_de_vida,
    estres_academico, apoyo_familiar, bienestar, asistencia,
    horas_estudio, interes_academico, rendimiento_academico,
    historial_academico, carga_laboral, beca, deudor,
    promedio_ponderado, edad, estado_seguimiento
) VALUES
    ('20230001', 'Ana Castillo Rojas', 'Ingeniería de Sistemas', 3,
     'Entre_6_8h', 'Moderado', 'Moderada', 'Moderado',
     'Moderado', 'Fuerte', 'Saludable', 'Constante',
     'De_1_3h', 'Muy_motivado', 'Previsto',
     'Entre_11_15', 'No_trabaja', 'Parcial', 'Sin_deuda',
     15.5, 20, 'En seguimiento'),

    ('20230002', 'Bruno Fernández', 'Administración', 2,
     'Menos_de_6h', 'Sedentario', 'Poco_saludable', 'Poco_saludable',
     'Severo', 'Escaso', 'En_riesgo', 'Irregular',
     'Menor_a_1h', 'Desmotivado', 'En_proceso',
     'Menor_a_11', 'Completa', 'No_tiene', 'Retraso_moderado',
     12.3, 22, 'Pendiente'),

    ('20230003', 'Carla Díaz', 'Derecho', 5,
     'Más_de_8h', 'Activa', 'Balanceada', 'Saludable',
     'Leve', 'Fuerte', 'Saludable', 'Constante',
     'Más_de_3h', 'Muy_motivado', 'Logro_destacado',
     'Mayor_a_15', 'No_trabaja', 'Completa', 'Sin_deuda',
     17.8, 21, 'En seguimiento');

-- Tabla para almacenar entrenamientos del modelo
CREATE TABLE IF NOT EXISTS entrenamientos_modelo (
    id SERIAL PRIMARY KEY,
    fecha_entrenamiento TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    num_estudiantes_entrenamiento INTEGER,
    precision_modelo DECIMAL(5,2),
    metricas_json JSONB,  -- Almacena métricas adicionales (accuracy, precision, recall, f1-score)
    modelo_version VARCHAR(50),
    observaciones TEXT,
    ruta_modelo VARCHAR(255)
);

-- Índice para consultas por fecha
CREATE INDEX IF NOT EXISTS idx_fecha_entrenamiento ON entrenamientos_modelo(fecha_entrenamiento);

-- Comentarios en las tablas
COMMENT ON TABLE estudiantes IS 'Tabla principal que contiene toda la información de estudiantes para el Sistema de Alerta Temprana';
COMMENT ON TABLE entrenamientos_modelo IS 'Historial de entrenamientos del modelo de predicción';
COMMENT ON COLUMN estudiantes.sueno_horas IS 'Horas promedio de sueño diario: Menos_de_6h, Entre_6_8h, Más_de_8h';
COMMENT ON COLUMN estudiantes.estres_academico IS 'Nivel de estrés académico: Leve, Moderado, Alto, Severo, Crítico';
COMMENT ON COLUMN estudiantes.riesgo_predicho IS 'Resultado del modelo: Sin_riesgo, Riesgo_leve, Riesgo_moderado, Riesgo_alto, Riesgo_critico';
COMMENT ON COLUMN estudiantes.cluster_asignado IS 'Cluster asignado por el algoritmo: 0, 1, 2';
