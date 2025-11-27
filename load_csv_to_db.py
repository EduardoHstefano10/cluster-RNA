"""
Script para cargar automáticamente el CSV de estudiantes a PostgreSQL
Convierte datos numéricos del CSV a registros en la base de datos
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import os
from dotenv import load_dotenv
from datetime import datetime
import numpy as np

# Cargar variables de entorno
load_dotenv()


def get_db_connection():
    """Obtener conexión a PostgreSQL"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'estudiantes_db'),
        user=os.getenv('DB_USER', 'cluster_user'),
        password=os.getenv('DB_PASSWORD', 'cluster_pass_2024')
    )


def convert_numeric_to_categorical(value, variable_name):
    """
    Convertir valores numéricos del CSV a categorías
    Basado en percentiles y rangos razonables
    """

    # Diccionario de conversiones según la variable
    # Estos valores son aproximados y deberían ajustarse según los datos reales

    if pd.isna(value):
        return None

    # Mapeamos las variables del CSV a variables categóricas del formulario
    # Por ahora, crearemos valores predeterminados basados en rangos

    # Las variables numéricas del CSV no tienen correspondencia directa
    # con las variables categóricas del formulario, así que usaremos valores por defecto
    # o crearemos lógica de conversión

    return None  # Retornar None por ahora, se usarán valores por defecto


def generate_categorical_defaults(row):
    """
    Generar valores categóricos por defecto basados en métricas del estudiante
    Usa lógica heurística para inferir comportamiento
    """

    promedio = row.get('Promedio_ponderado', 14)
    asistencia = row.get('Asistencia', 85)
    cursos_desaprobados = row.get('Cursos_desaprobados', 0)
    horas_estudio = row.get('Horas_estudio_semana', 15)
    horas_trabajo = row.get('Horas_trabajo_semana', 0)

    # Lógica de inferencia
    categoricas = {}

    # Sueño (basado en horas de estudio y trabajo)
    total_horas = horas_estudio + horas_trabajo
    if total_horas > 45:
        categoricas['sueno_horas'] = 'Menos_de_6h'
    elif total_horas > 30:
        categoricas['sueno_horas'] = 'Entre_6_8h'
    else:
        categoricas['sueno_horas'] = 'Más_de_8h'

    # Actividad física (basado en carga laboral y de estudio)
    if horas_trabajo > 30 or horas_estudio > 35:
        categoricas['actividad_fisica'] = 'Sedentario'
    elif horas_trabajo > 15 or horas_estudio > 25:
        categoricas['actividad_fisica'] = 'Moderado'
    else:
        categoricas['actividad_fisica'] = 'Activa'

    # Alimentación (inferido del promedio y asistencia)
    if promedio >= 16 and asistencia >= 90:
        categoricas['alimentacion'] = 'Balanceada'
    elif promedio >= 13 and asistencia >= 80:
        categoricas['alimentacion'] = 'Moderada'
    else:
        categoricas['alimentacion'] = 'Poco_saludable'

    # Estilo de vida (combinación de factores)
    if promedio >= 16 and asistencia >= 90 and cursos_desaprobados == 0:
        categoricas['estilo_de_vida'] = 'Saludable'
    elif promedio >= 13 and asistencia >= 75:
        categoricas['estilo_de_vida'] = 'Moderado'
    else:
        categoricas['estilo_de_vida'] = 'Poco_saludable'

    # Estrés académico (basado en cursos desaprobados y carga)
    if cursos_desaprobados >= 3 or promedio < 11:
        categoricas['estres_academico'] = 'Crítico'
    elif cursos_desaprobados >= 2 or promedio < 12:
        categoricas['estres_academico'] = 'Severo'
    elif cursos_desaprobados >= 1 or promedio < 13:
        categoricas['estres_academico'] = 'Alto'
    elif promedio < 15:
        categoricas['estres_academico'] = 'Moderado'
    else:
        categoricas['estres_academico'] = 'Leve'

    # Apoyo familiar (inferido de otros factores)
    if promedio >= 16 and asistencia >= 90:
        categoricas['apoyo_familiar'] = 'Fuerte'
    elif promedio >= 14 and asistencia >= 80:
        categoricas['apoyo_familiar'] = 'Moderado'
    elif promedio >= 12:
        categoricas['apoyo_familiar'] = 'Escaso'
    else:
        categoricas['apoyo_familiar'] = 'Nulo'

    # Bienestar (basado en múltiples factores)
    if promedio >= 16 and asistencia >= 90 and cursos_desaprobados == 0:
        categoricas['bienestar'] = 'Saludable'
    elif promedio >= 13 and asistencia >= 75:
        categoricas['bienestar'] = 'Moderado'
    else:
        categoricas['bienestar'] = 'En_riesgo'

    # Asistencia categórica (basada en porcentaje)
    if asistencia >= 95:
        categoricas['asistencia'] = 'Constante'
    elif asistencia >= 85:
        categoricas['asistencia'] = 'Frecuente'
    elif asistencia >= 70:
        categoricas['asistencia'] = 'Irregular'
    else:
        categoricas['asistencia'] = 'Nula'

    # Horas de estudio categóricas
    if horas_estudio >= 20:
        categoricas['horas_estudio'] = 'Más_de_3h'
    elif horas_estudio >= 7:
        categoricas['horas_estudio'] = 'De_1_3h'
    else:
        categoricas['horas_estudio'] = 'Menor_a_1h'

    # Interés académico (basado en promedio y asistencia)
    if promedio >= 17 and asistencia >= 90:
        categoricas['interes_academico'] = 'Muy_motivado'
    elif promedio >= 13 and asistencia >= 75:
        categoricas['interes_academico'] = 'Regular'
    else:
        categoricas['interes_academico'] = 'Desmotivado'

    # Rendimiento académico
    if promedio >= 18:
        categoricas['rendimiento_academico'] = 'Logro_destacado'
    elif promedio >= 15:
        categoricas['rendimiento_academico'] = 'Previsto'
    elif promedio >= 12:
        categoricas['rendimiento_academico'] = 'En_proceso'
    else:
        categoricas['rendimiento_academico'] = 'En_inicio'

    # Historial académico (basado en nota promedio)
    nota_prom = row.get('Nota_promedio', 14)
    if nota_prom >= 16:
        categoricas['historial_academico'] = 'Mayor_a_15'
    elif nota_prom >= 11:
        categoricas['historial_academico'] = 'Entre_11_15'
    else:
        categoricas['historial_academico'] = 'Menor_a_11'

    # Carga laboral
    if horas_trabajo >= 35:
        categoricas['carga_laboral'] = 'Completa'
    elif horas_trabajo >= 15:
        categoricas['carga_laboral'] = 'Parcial'
    else:
        categoricas['carga_laboral'] = 'No_trabaja'

    # Beca (basado en promedio)
    if promedio >= 17:
        categoricas['beca'] = 'Completa'
    elif promedio >= 15:
        categoricas['beca'] = 'Parcial'
    else:
        categoricas['beca'] = 'No_tiene'

    # Estado de deudor (aleatorio pero coherente con situación)
    if horas_trabajo >= 30:
        categoricas['deudor'] = 'Retraso_crítico'
    elif horas_trabajo >= 20:
        categoricas['deudor'] = 'Retraso_moderado'
    elif horas_trabajo >= 10:
        categoricas['deudor'] = 'Retraso_leve'
    else:
        categoricas['deudor'] = 'Sin_deuda'

    return categoricas


def load_csv_to_database(csv_path='data/estudiantes_data.csv', batch_size=100):
    """
    Cargar datos del CSV a PostgreSQL
    """

    print("🚀 Iniciando carga de datos del CSV a PostgreSQL...")

    # Leer CSV
    print(f"📂 Leyendo archivo: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Archivo leído: {len(df)} registros encontrados")

    # Conectar a base de datos
    print("🔌 Conectando a PostgreSQL...")
    conn = get_db_connection()
    cursor = conn.cursor()
    print("✅ Conexión establecida")

    # Limpiar tabla existente (opcional - comentar si quieres mantener datos)
    print("🧹 Limpiando tabla de estudiantes...")
    cursor.execute("DELETE FROM estudiantes WHERE codigo LIKE '2015%' OR codigo LIKE '2016%' OR codigo LIKE 'CSV%'")
    conn.commit()
    print("✅ Tabla limpiada")

    # Preparar datos para inserción
    carreras = ['Ingeniería', 'Administración', 'Derecho', 'Medicina',
                'Psicología', 'Arquitectura', 'Educación', 'Contabilidad']

    # Query de inserción
    insert_query = """
    INSERT INTO estudiantes (
        codigo, nombre, carrera, ciclo, edad,
        sueno_horas, actividad_fisica, alimentacion, estilo_de_vida,
        estres_academico, apoyo_familiar, bienestar, asistencia,
        horas_estudio, interes_academico, rendimiento_academico,
        historial_academico, carga_laboral, beca, deudor,
        promedio_ponderado, creditos_matriculados, porcentaje_creditos_aprobados,
        cursos_desaprobados, asistencia_porcentaje, retiros_cursos,
        horas_trabajo_semana, anio_ingreso, numero_ciclos_academicos,
        cursos_matriculados_ciclo, horas_estudio_semana, indice_regularidad,
        intentos_aprobacion_curso, nota_promedio,
        notas_tutor, estado_seguimiento
    ) VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s,
        %s, %s
    )
    ON CONFLICT (codigo) DO UPDATE SET
        ultima_actualizacion = CURRENT_TIMESTAMP
    """

    # Preparar lotes de datos
    batch_data = []
    total_inserted = 0

    print(f"📝 Procesando {len(df)} registros...")

    for idx, row in df.iterrows():
        # Generar código único
        codigo = f"CSV{str(idx + 1).zfill(6)}"
        nombre = f"Estudiante CSV {idx + 1}"
        carrera = carreras[idx % len(carreras)]
        ciclo = int(row.get('Numero_ciclos_academicos', 5))
        edad = int(row.get('Edad', 20))

        # Generar categorías
        categoricas = generate_categorical_defaults(row)

        # Preparar valores
        values = (
            codigo,
            nombre,
            carrera,
            min(ciclo, 12),  # Limitar a 12 ciclos
            edad,
            # Variables categóricas
            categoricas['sueno_horas'],
            categoricas['actividad_fisica'],
            categoricas['alimentacion'],
            categoricas['estilo_de_vida'],
            categoricas['estres_academico'],
            categoricas['apoyo_familiar'],
            categoricas['bienestar'],
            categoricas['asistencia'],
            categoricas['horas_estudio'],
            categoricas['interes_academico'],
            categoricas['rendimiento_academico'],
            categoricas['historial_academico'],
            categoricas['carga_laboral'],
            categoricas['beca'],
            categoricas['deudor'],
            # Variables numéricas
            float(row.get('Promedio_ponderado', 14)),
            int(row.get('Creditos_matriculados', 20)),
            float(row.get('Porcentaje_creditos_aprobados', 75)),
            int(abs(row.get('Cursos_desaprobados', 0))),
            float(row.get('Asistencia', 85)),
            int(abs(row.get('Retiros_cursos', 0))),
            float(row.get('Horas_trabajo_semana', 0)),
            int(row.get('Anio_ingreso', 2020)),
            int(row.get('Numero_ciclos_academicos', 5)),
            int(row.get('Cursos_matriculados_ciclo', 6)),
            float(row.get('Horas_estudio_semana', 15)),
            float(row.get('indice_regularidad', 65)),
            int(abs(row.get('Intentos_aprobacion_curso', 1))),
            float(row.get('Nota_promedio', 14)),
            # Metadatos
            f"Registro importado automáticamente desde CSV el {datetime.now().strftime('%Y-%m-%d')}",
            'Pendiente'
        )

        batch_data.append(values)

        # Insertar en lotes
        if len(batch_data) >= batch_size:
            execute_batch(cursor, insert_query, batch_data)
            conn.commit()
            total_inserted += len(batch_data)
            print(f"✅ Insertados {total_inserted}/{len(df)} registros...")
            batch_data = []

    # Insertar registros restantes
    if batch_data:
        execute_batch(cursor, insert_query, batch_data)
        conn.commit()
        total_inserted += len(batch_data)

    print(f"\n🎉 Carga completada exitosamente!")
    print(f"📊 Total de registros insertados: {total_inserted}")

    # Estadísticas
    cursor.execute("SELECT COUNT(*) FROM estudiantes")
    total = cursor.fetchone()[0]
    print(f"📈 Total de estudiantes en la base de datos: {total}")

    # Cerrar conexión
    cursor.close()
    conn.close()
    print("✅ Conexión cerrada")


if __name__ == "__main__":
    try:
        load_csv_to_database()
    except Exception as e:
        print(f"❌ Error durante la carga: {e}")
        import traceback
        traceback.print_exc()
