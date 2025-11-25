"""
Módulo de conexión y operaciones con PostgreSQL
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


class DatabaseConnection:
    """Gestor de conexión a PostgreSQL"""

    def __init__(self):
        self.config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'estudiantes_db'),
            'user': os.getenv('DB_USER', 'cluster_user'),
            'password': os.getenv('DB_PASSWORD', 'cluster_pass_2024')
        }
        self.conn = None
        self.cursor = None

    def connect(self):
        """Establecer conexión a la base de datos"""
        try:
            self.conn = psycopg2.connect(**self.config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            print("✅ Conexión a PostgreSQL establecida exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error al conectar a PostgreSQL: {e}")
            return False

    def close(self):
        """Cerrar conexión"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("✅ Conexión a PostgreSQL cerrada")

    def execute_query(self, query: str, params: tuple = None):
        """Ejecutar una consulta SQL"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            self.conn.commit()
            return True
        except Exception as e:
            print(f"❌ Error al ejecutar query: {e}")
            self.conn.rollback()
            return False

    def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        """Obtener todos los resultados de una consulta"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"❌ Error al obtener datos: {e}")
            return []

    def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        """Obtener un solo resultado"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.fetchone()
        except Exception as e:
            print(f"❌ Error al obtener dato: {e}")
            return None


class EstudiantesDB:
    """Operaciones específicas para la tabla de estudiantes"""

    def __init__(self):
        self.db = DatabaseConnection()
        self.db.connect()

    def get_all_students(self, limit: int = None) -> pd.DataFrame:
        """Obtener todos los estudiantes como DataFrame"""
        query = "SELECT * FROM estudiantes"
        if limit:
            query += f" LIMIT {limit}"

        data = self.db.fetch_all(query)
        return pd.DataFrame(data)

    def get_student_by_codigo(self, codigo: str) -> Optional[Dict]:
        """Obtener estudiante por código"""
        query = "SELECT * FROM estudiantes WHERE codigo = %s"
        return self.db.fetch_one(query, (codigo,))

    def get_training_data(self) -> pd.DataFrame:
        """
        Obtener datos para entrenamiento del modelo
        Convierte variables categóricas a numéricas
        """
        query = """
        SELECT
            sueno_horas,
            actividad_fisica,
            alimentacion,
            estilo_de_vida,
            estres_academico,
            apoyo_familiar,
            bienestar,
            asistencia,
            horas_estudio,
            interes_academico,
            rendimiento_academico,
            historial_academico,
            carga_laboral,
            beca,
            deudor,
            riesgo_predicho
        FROM estudiantes
        WHERE riesgo_predicho IS NOT NULL
        """
        data = self.db.fetch_all(query)
        return pd.DataFrame(data)

    def insert_student(self, student_data: Dict) -> bool:
        """Insertar nuevo estudiante"""
        query = """
        INSERT INTO estudiantes (
            codigo, nombre, carrera, ciclo,
            sueno_horas, actividad_fisica, alimentacion, estilo_de_vida,
            estres_academico, apoyo_familiar, bienestar, asistencia,
            horas_estudio, interes_academico, rendimiento_academico,
            historial_academico, carga_laboral, beca, deudor,
            promedio_ponderado, edad, notas_tutor
        ) VALUES (
            %(codigo)s, %(nombre)s, %(carrera)s, %(ciclo)s,
            %(sueno_horas)s, %(actividad_fisica)s, %(alimentacion)s, %(estilo_de_vida)s,
            %(estres_academico)s, %(apoyo_familiar)s, %(bienestar)s, %(asistencia)s,
            %(horas_estudio)s, %(interes_academico)s, %(rendimiento_academico)s,
            %(historial_academico)s, %(carga_laboral)s, %(beca)s, %(deudor)s,
            %(promedio_ponderado)s, %(edad)s, %(notas_tutor)s
        )
        """
        try:
            self.db.cursor.execute(query, student_data)
            self.db.conn.commit()
            return True
        except Exception as e:
            print(f"❌ Error al insertar estudiante: {e}")
            self.db.conn.rollback()
            return False

    def update_prediction(self, codigo: str, prediction_data: Dict) -> bool:
        """Actualizar predicción de un estudiante"""
        query = """
        UPDATE estudiantes
        SET riesgo_predicho = %(riesgo_predicho)s,
            cluster_asignado = %(cluster_asignado)s,
            probabilidad_desercion = %(probabilidad_desercion)s
        WHERE codigo = %(codigo)s
        """
        prediction_data['codigo'] = codigo

        try:
            self.db.cursor.execute(query, prediction_data)
            self.db.conn.commit()
            return True
        except Exception as e:
            print(f"❌ Error al actualizar predicción: {e}")
            self.db.conn.rollback()
            return False

    def get_statistics(self) -> Dict:
        """Obtener estadísticas generales"""
        stats = {}

        # Total de estudiantes
        query = "SELECT COUNT(*) as total FROM estudiantes"
        result = self.db.fetch_one(query)
        stats['total_estudiantes'] = result['total'] if result else 0

        # Estudiantes en alto riesgo
        query = """
        SELECT COUNT(*) as alto_riesgo
        FROM estudiantes
        WHERE riesgo_predicho IN ('Riesgo_alto', 'Riesgo_critico')
        """
        result = self.db.fetch_one(query)
        stats['estudiantes_alto_riesgo'] = result['alto_riesgo'] if result else 0

        # Distribución por cluster
        query = """
        SELECT cluster_asignado, COUNT(*) as count
        FROM estudiantes
        WHERE cluster_asignado IS NOT NULL
        GROUP BY cluster_asignado
        """
        clusters = self.db.fetch_all(query)
        stats['distribucion_clusters'] = {c['cluster_asignado']: c['count'] for c in clusters}

        return stats

    def close(self):
        """Cerrar conexión"""
        self.db.close()


# Funciones de utilidad
def test_connection():
    """Probar conexión a la base de datos"""
    db = DatabaseConnection()
    if db.connect():
        print("✅ Conexión exitosa a PostgreSQL")
        db.close()
        return True
    else:
        print("❌ No se pudo conectar a PostgreSQL")
        return False


if __name__ == "__main__":
    # Test de conexión
    test_connection()

    # Test de operaciones
    db = EstudiantesDB()

    # Obtener estadísticas
    stats = db.get_statistics()
    print(f"\n📊 Estadísticas:")
    print(f"  Total estudiantes: {stats.get('total_estudiantes', 0)}")
    print(f"  Alto riesgo: {stats.get('estudiantes_alto_riesgo', 0)}")
    print(f"  Clusters: {stats.get('distribucion_clusters', {})}")

    # Obtener todos los estudiantes
    df = db.get_all_students(limit=5)
    print(f"\n👥 Primeros 5 estudiantes:")
    print(df[['codigo', 'nombre', 'carrera']].to_string())

    db.close()
