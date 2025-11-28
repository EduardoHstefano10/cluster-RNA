"""
Script para insertar datos de entrenamiento de referencia en la base de datos
Simula entrenamientos históricos del modelo con diferentes métricas y configuraciones
"""

import psycopg2
from psycopg2.extras import execute_batch
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import random

# Cargar variables de entorno
load_dotenv()


def get_db_connection():
    """Obtener conexión a PostgreSQL"""
    database_url = os.getenv('DATABASE_URL') or os.getenv('DATABASE_PUBLIC_URL')

    if database_url:
        print("🔗 Usando DATABASE_URL para conectar...")
        return psycopg2.connect(database_url)

    print("🔗 Usando variables individuales para conectar...")
    return psycopg2.connect(
        host=os.getenv('PGHOST', os.getenv('DB_HOST', 'localhost')),
        port=int(os.getenv('PGPORT', os.getenv('DB_PORT', 5432))),
        database=os.getenv('PGDATABASE', os.getenv('DB_NAME', 'railway')),
        user=os.getenv('PGUSER', os.getenv('DB_USER', 'postgres')),
        password=os.getenv('PGPASSWORD', os.getenv('DB_PASSWORD', ''))
    )


def generate_training_data():
    """
    Generar datos de entrenamiento de referencia
    Simula entrenamientos progresivos del modelo mejorando con el tiempo
    """

    # Fecha actual
    current_date = datetime.now()

    # Lista de entrenamientos históricos
    trainings = []

    # Configuraciones de entrenamientos progresivos (de más antiguos a más recientes)
    training_configs = [
        # Entrenamientos iniciales (hace 6 meses) - menor precisión
        {
            'days_ago': 180,
            'num_students': 150,
            'version': 'v1.0-baseline',
            'config': 'Red neuronal simple (64,32)',
            'train_acc': 0.72,
            'test_acc': 0.68,
            'variation': 0.02
        },
        {
            'days_ago': 175,
            'num_students': 200,
            'version': 'v1.1-improved',
            'config': 'Red neuronal (128,64)',
            'train_acc': 0.75,
            'test_acc': 0.71,
            'variation': 0.02
        },
        # Entrenamientos medios (hace 4 meses)
        {
            'days_ago': 120,
            'num_students': 300,
            'version': 'v1.2-optimized',
            'config': 'Red neuronal (128,64,32) con PCA',
            'train_acc': 0.78,
            'test_acc': 0.74,
            'variation': 0.02
        },
        {
            'days_ago': 110,
            'num_students': 350,
            'version': 'v1.3-tuned',
            'config': 'Red neuronal (128,64,32) con early stopping',
            'train_acc': 0.81,
            'test_acc': 0.77,
            'variation': 0.02
        },
        # Entrenamientos recientes (hace 2 meses)
        {
            'days_ago': 60,
            'num_students': 450,
            'version': 'v2.0-categorical',
            'config': 'Modelo con variables categóricas y clustering',
            'train_acc': 0.84,
            'test_acc': 0.80,
            'variation': 0.02
        },
        {
            'days_ago': 50,
            'num_students': 500,
            'version': 'v2.1-enhanced',
            'config': 'Modelo mejorado con más datos de entrenamiento',
            'train_acc': 0.86,
            'test_acc': 0.82,
            'variation': 0.015
        },
        # Entrenamientos muy recientes (último mes)
        {
            'days_ago': 30,
            'num_students': 600,
            'version': 'v2.2-stable',
            'config': 'Modelo estable con validación cruzada',
            'train_acc': 0.88,
            'test_acc': 0.84,
            'variation': 0.015
        },
        {
            'days_ago': 20,
            'num_students': 650,
            'version': 'v2.3-optimized',
            'config': 'Optimización de hiperparámetros',
            'train_acc': 0.89,
            'test_acc': 0.86,
            'variation': 0.01
        },
        {
            'days_ago': 10,
            'num_students': 700,
            'version': 'v2.4-production',
            'config': 'Modelo en producción con datos actualizados',
            'train_acc': 0.90,
            'test_acc': 0.87,
            'variation': 0.01
        },
        # Entrenamiento más reciente (hace 3 días)
        {
            'days_ago': 3,
            'num_students': 750,
            'version': 'v2.5-latest',
            'config': 'Última versión con ajustes finos',
            'train_acc': 0.91,
            'test_acc': 0.88,
            'variation': 0.01
        }
    ]

    # Generar entrenamientos
    for config in training_configs:
        # Calcular fecha del entrenamiento
        training_date = current_date - timedelta(days=config['days_ago'])

        # Agregar variación aleatoria a las métricas
        variation = config['variation']
        train_acc = min(0.99, max(0.50, config['train_acc'] + random.uniform(-variation, variation)))
        test_acc = min(0.99, max(0.50, config['test_acc'] + random.uniform(-variation, variation)))

        # Calcular métricas derivadas
        # Precision, recall y f1-score simulados basados en accuracy
        precision = min(0.99, test_acc + random.uniform(-0.02, 0.03))
        recall = min(0.99, test_acc + random.uniform(-0.03, 0.02))
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Crear objeto de métricas
        metricas = {
            'train_accuracy': round(train_acc, 4),
            'test_accuracy': round(test_acc, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4),
            'pca_components': random.randint(8, 15),
            'training_samples': config['num_students'],
            'configuration': config['config']
        }

        # Crear observaciones descriptivas
        observaciones = f"""Entrenamiento del modelo {config['version']}

Configuración: {config['config']}
Número de estudiantes: {config['num_students']}
Fecha: {training_date.strftime('%Y-%m-%d %H:%M')}

Resultados:
- Train Accuracy: {train_acc:.2%}
- Test Accuracy: {test_acc:.2%}
- Precisión: {precision:.2%}
- Recall: {recall:.2%}
- F1-Score: {f1_score:.2%}

Estado: {"✅ Modelo estable y en producción" if config['days_ago'] < 15 else "✓ Entrenamiento histórico completado"}
"""

        # Ruta del modelo
        ruta_modelo = f"models/{config['version']}/model_{training_date.strftime('%Y%m%d')}.pkl"

        training_record = {
            'fecha': training_date,
            'num_estudiantes': config['num_students'],
            'precision': round(test_acc * 100, 2),  # Convertir a porcentaje
            'metricas': json.dumps(metricas),
            'version': config['version'],
            'observaciones': observaciones,
            'ruta': ruta_modelo
        }

        trainings.append(training_record)

    return trainings


def insert_training_data():
    """
    Insertar datos de entrenamiento en la base de datos
    """
    print("🚀 Iniciando inserción de datos de entrenamiento de referencia...\n")

    # Generar datos
    print("📊 Generando datos de entrenamiento de referencia...")
    trainings = generate_training_data()
    print(f"✅ Generados {len(trainings)} registros de entrenamiento\n")

    # Conectar a base de datos
    print("🔌 Conectando a PostgreSQL...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        print("✅ Conexión establecida\n")
    except Exception as e:
        print(f"❌ Error al conectar: {e}")
        return False

    # Limpiar tabla existente (opcional)
    print("🧹 Limpiando registros anteriores...")
    try:
        cursor.execute("DELETE FROM entrenamientos_modelo")
        conn.commit()
        print("✅ Tabla limpiada\n")
    except Exception as e:
        print(f"⚠️  No se pudo limpiar la tabla: {e}\n")
        conn.rollback()

    # Query de inserción
    insert_query = """
    INSERT INTO entrenamientos_modelo (
        fecha_entrenamiento,
        num_estudiantes_entrenamiento,
        precision_modelo,
        metricas_json,
        modelo_version,
        observaciones,
        ruta_modelo
    ) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
    """

    # Insertar datos
    print("📝 Insertando datos de entrenamiento...\n")
    try:
        for i, training in enumerate(trainings, 1):
            cursor.execute(insert_query, (
                training['fecha'],
                training['num_estudiantes'],
                training['precision'],
                training['metricas'],
                training['version'],
                training['observaciones'],
                training['ruta']
            ))

            # Mostrar progreso
            metricas = json.loads(training['metricas'])
            print(f"  [{i}/{len(trainings)}] {training['version']}")
            print(f"       📅 {training['fecha'].strftime('%Y-%m-%d')}")
            print(f"       👥 {training['num_estudiantes']} estudiantes")
            print(f"       📈 Precisión: {training['precision']:.2f}%")
            print(f"       ✓ Train: {metricas['train_accuracy']:.2%} | Test: {metricas['test_accuracy']:.2%}")
            print()

        conn.commit()
        print("✅ Todos los datos insertados exitosamente!\n")

    except Exception as e:
        print(f"❌ Error al insertar datos: {e}")
        conn.rollback()
        cursor.close()
        conn.close()
        return False

    # Verificar inserción
    print("🔍 Verificando datos insertados...")
    try:
        cursor.execute("SELECT COUNT(*) FROM entrenamientos_modelo")
        count = cursor.fetchone()[0]
        print(f"✅ Total de entrenamientos en la base de datos: {count}\n")

        # Mostrar estadísticas
        cursor.execute("""
            SELECT
                MIN(precision_modelo) as min_precision,
                MAX(precision_modelo) as max_precision,
                AVG(precision_modelo) as avg_precision,
                MIN(num_estudiantes_entrenamiento) as min_students,
                MAX(num_estudiantes_entrenamiento) as max_students
            FROM entrenamientos_modelo
        """)
        stats = cursor.fetchone()

        print("📊 Estadísticas de entrenamientos:")
        print(f"   Precisión mínima: {stats[0]:.2f}%")
        print(f"   Precisión máxima: {stats[1]:.2f}%")
        print(f"   Precisión promedio: {stats[2]:.2f}%")
        print(f"   Estudiantes (min-max): {stats[3]} - {stats[4]}")
        print()

    except Exception as e:
        print(f"⚠️  Error al verificar datos: {e}\n")

    # Cerrar conexión
    cursor.close()
    conn.close()
    print("✅ Conexión cerrada")
    print("\n" + "="*60)
    print("🎉 ¡Datos de entrenamiento insertados exitosamente!")
    print("="*60)
    print("\n💡 Ahora puedes ver los entrenamientos en la interfaz web")
    print("   en la página de 'Comparación de Modelos'\n")

    return True


if __name__ == "__main__":
    try:
        success = insert_training_data()
        if not success:
            print("\n❌ Hubo errores durante la inserción")
            exit(1)
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
