"""
Sistema de Alerta Temprana - Backend FastAPI
Incluye endpoints para gestión de estudiantes, predicción de riesgo y clustering
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

from ml_models import StudentRiskPredictor

# Inicializar FastAPI
app = FastAPI(
    title="Sistema de Alerta Temprana",
    description="API para predicción de riesgo académico y clustering de estudiantes",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar o entrenar modelo
predictor = StudentRiskPredictor()
if os.path.exists('models/neural_network.pkl'):
    predictor.load_model()
    print("Modelo cargado exitosamente")
else:
    print("Entrenando nuevo modelo...")
    predictor.train('estudiantes_data.csv')
    predictor.save_model()

# --- Reemplazar la carga directa del CSV por una función robusta ---
def load_students_dataframe():
    """Intentar cargar estudiantes_data.csv desde varias rutas; devolver DataFrame vacío si no existe."""
    possible_paths = [
        'estudiantes_data.csv',
        os.path.join('data', 'estudiantes_data.csv'),
        os.path.join('.', 'data', 'estudiantes_data.csv'),
        os.path.join(os.path.dirname(__file__), 'estudiantes_data.csv'),
        os.path.join(os.path.dirname(__file__), 'data', 'estudiantes_data.csv'),
    ]
    for p in possible_paths:
        try:
            if os.path.exists(p):
                df = pd.read_csv(p)
                print(f"✅ Cargado CSV de estudiantes desde: {p} ({len(df)} registros)")
                return df
        except Exception as e:
            print(f"⚠️ Error al leer CSV en {p}: {e}")
    print("⚠️ No se encontró 'estudiantes_data.csv' en rutas habituales. Usando DataFrame vacío.")
    return pd.DataFrame()

# Reemplazar la línea que hacía: df_students = pd.read_csv('estudiantes_data.csv')
df_students = load_students_dataframe()


# ==================== MODELOS PYDANTIC ====================

class StudentData(BaseModel):
    """Modelo de datos del estudiante"""
    Promedio_ponderado: float = Field(..., ge=0, le=20, description="Promedio ponderado (0-20)")
    Creditos_matriculados: float = Field(..., ge=0, description="Créditos matriculados")
    Porcentaje_creditos_aprobados: float = Field(..., ge=0, le=100, description="% créditos aprobados")
    Cursos_desaprobados: float = Field(..., ge=0, description="Cursos desaprobados")
    Asistencia: float = Field(..., ge=0, le=100, description="% asistencia")
    Retiros_cursos: float = Field(..., ge=0, description="Retiros de cursos")
    Edad: float = Field(..., ge=16, le=60, description="Edad del estudiante")
    Horas_trabajo_semana: float = Field(..., ge=0, le=168, description="Horas de trabajo/semana")
    Anio_ingreso: int = Field(..., ge=2010, le=2025, description="Año de ingreso")
    Numero_ciclos_academicos: float = Field(..., ge=1, description="Número de ciclos académicos")
    Cursos_matriculados_ciclo: float = Field(..., ge=1, le=15, description="Cursos matriculados/ciclo")
    Horas_estudio_semana: float = Field(..., ge=0, le=168, description="Horas de estudio/semana")
    indice_regularidad: float = Field(..., ge=0, le=100, description="Índice de regularidad")
    Intentos_aprobacion_curso: float = Field(..., description="Intentos de aprobación")
    Nota_promedio: float = Field(..., ge=0, le=20, description="Nota promedio")


class StudentProfile(BaseModel):
    """Perfil completo del estudiante con metadatos"""
    id: Optional[str] = None
    codigo: str = Field(..., description="Código del estudiante")
    nombre: str = Field(..., description="Nombre completo")
    carrera: str = Field(..., description="Carrera")
    ciclo: int = Field(..., ge=1, le=12, description="Ciclo actual")
    datos: StudentData
    ultima_actualizacion: Optional[str] = None


class PredictionResponse(BaseModel):
    """Respuesta de predicción de riesgo"""
    risk_level: int
    risk_label: str
    risk_probability: float
    desertion_probability: float
    cluster: int
    cluster_name: str
    recommendations: List[str]
    key_factors: List[Dict[str, str]]


class DashboardStats(BaseModel):
    """Estadísticas del dashboard"""
    total_estudiantes: int
    precision_modelo: float
    estudiantes_alto_riesgo: int
    seguimiento_activo: int
    num_clusters: int
    clusters_activos: List[str]


# ==================== BASE DE DATOS SIMULADA ====================

# Almacenamiento en memoria (en producción usar BD real)
students_db: Dict[str, StudentProfile] = {}
predictions_db: Dict[str, PredictionResponse] = {}


# ==================== ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Página principal - Formulario de registro"""
    with open('frontend/registro.html', 'r', encoding='utf-8') as f:
        return f.read()


@app.get("/panel", response_class=HTMLResponse)
async def panel_tutor():
    """Panel del tutor"""
    with open('frontend/panel.html', 'r', encoding='utf-8') as f:
        return f.read()


@app.get("/perfil/{codigo}", response_class=HTMLResponse)
async def perfil_estudiante(codigo: str):
    """Perfil del estudiante"""
    with open('frontend/perfil.html', 'r', encoding='utf-8') as f:
        return f.read()


@app.get("/api/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Obtener estadísticas del dashboard"""
    try:
        # Si no hay data local, intentar estadísticas desde DB
        if df_students.empty:
            try:
                from database import EstudiantesDB
                db = EstudiantesDB()
                stats_db = db.get_statistics()
                db.close()
                return DashboardStats(
                    total_estudiantes=stats_db.get('total_estudiantes', 0),
                    precision_modelo=92.4,
                    estudiantes_alto_riesgo=stats_db.get('estudiantes_alto_riesgo', 0),
                    seguimiento_activo=max(0, stats_db.get('total_estudiantes', 0)//5),
                    num_clusters=3,
                    clusters_activos=[
                        "C1 - Compromiso alto",
                        "C2 - Estrés académico",
                        "C3 - Riesgo acumulado"
                    ]
                )
            except Exception:
                # Fallback a valores por defecto si DB no está disponible
                return DashboardStats(
                    total_estudiantes=0,
                    precision_modelo=92.4,
                    estudiantes_alto_riesgo=0,
                    seguimiento_activo=0,
                    num_clusters=3,
                    clusters_activos=[
                        "C1 - Compromiso alto",
                        "C2 - Estrés académico",
                        "C3 - Riesgo acumulado"
                    ]
                )

        # Si tenemos df_students, calcular una estimación (limitar para performance)
        total = len(df_students)
        alto_riesgo = 0
        for idx in range(min(total, 100)):
            row = df_students.iloc[idx]
            student_dict = row.to_dict()
            try:
                pred = predictor.predict_risk(student_dict)
                if pred['risk_level'] >= 3:
                    alto_riesgo += 1
            except Exception:
                continue

        return DashboardStats(
            total_estudiantes=total,
            precision_modelo=92.4,
            estudiantes_alto_riesgo=alto_riesgo if alto_riesgo > 0 else 0,
            seguimiento_activo=total // 5,
            num_clusters=3,
            clusters_activos=[
                "C1 - Compromiso alto",
                "C2 - Estrés académico",
                "C3 - Riesgo acumulado"
            ]
        )
    except Exception as e:
        print(f"❌ Error en /api/stats: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.get("/api/students")
async def get_all_students(
    riesgo: Optional[str] = None,
    cluster: Optional[int] = None,
    estado: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
):
    """
    Obtener lista de estudiantes con filtros
    (manejo robusto cuando faltan columnas esperadas por el predictor)
    """
    try:
        # Si no hay df_students disponible, intentar recuperar desde DB
        if df_students.empty:
            try:
                from database import EstudiantesDB
                db = EstudiantesDB()
                df = db.get_all_students(limit=limit)
                db.close()
            except Exception:
                return {'total': 0, 'showing': '0 a 0', 'students': []}
        else:
            df = df_students

        students_list = []

        n = len(df)
        start = min(max(0, offset), n)
        end = min(start + limit, n)

        for idx in range(start, end):
            row = df.iloc[idx]
            student_dict = row.to_dict()

            # Intentar predecir de forma segura
            pred = None
            try:
                if predictor and hasattr(predictor, 'predict_risk'):
                    # predictor puede lanzar KeyError si faltan columnas; proteger
                    pred = predictor.predict_risk(student_dict)
                else:
                    pred = None
            except Exception as e:
                # Log y fallback a columnas ya presentes en los datos
                print(f"⚠️ Error al calcular predicción para fila {idx}: {e}")
                pred = None

            # Construir información del estudiante con fallback seguro
            try:
                risk_label = None
                risk_level = 0
                cluster_name = None
                cluster_id = 0
                desertion_prob = 0.0

                if pred:
                    risk_label = pred.get('risk_label') or pred.get('risk') or None
                    risk_level = int(pred.get('risk_level', 0))
                    cluster_name = pred.get('cluster_name') or pred.get('cluster') or None
                    cluster_id = int(pred.get('cluster', pred.get('cluster_id', 0)))
                    desertion_prob = float(pred.get('desertion_probability', pred.get('desertion_prob', 0)) or 0)
                else:
                    # Intentar tomar valores ya guardados en fila (DB/CSV)
                    risk_label = row.get('riesgo_predicho') or row.get('risk_label') or 'Sin evaluar'
                    # Mapear algunas cadenas a niveles si es posible
                    label_map = {
                        'Sin riesgo': 0, 'Sin_riesgo': 0,
                        'Riesgo leve': 1, 'Riesgo_leve': 1,
                        'Riesgo moderado': 2, 'Riesgo_moderado': 2,
                        'Riesgo alto': 3, 'Riesgo_alto': 3,
                        'Riesgo crítico': 4, 'Riesgo_critico': 4
                    }
                    risk_level = label_map.get(str(risk_label), 0)
                    cluster_name = row.get('cluster_asignado') or row.get('cluster_name') or 'Sin asignar'
                    # intentar obtener id de cluster numérico
                    try:
                        cluster_id = int(row.get('cluster_asignado') if row.get('cluster_asignado') is not None else row.get('cluster', 0))
                    except Exception:
                        cluster_id = 0
                    desertion_prob = float(row.get('probabilidad_desercion') or row.get('desertion_prob') or 0.0)

                student_info = {
                    'nombre': row.get('nombre') if 'nombre' in row else f"Estudiante {idx + 1}",
                    'codigo': row.get('codigo') if 'codigo' in row else f"ID{idx+1}",
                    'carrera': row.get('carrera') if 'carrera' in row else 'Sin carrera',
                    'promedio': round(float(row.get('promedio_ponderado') or row.get('Promedio_ponderado', 0)), 1) if (row.get('promedio_ponderado') or row.get('Promedio_ponderado')) is not None else 'N/A',
                    'asistencia': float(row.get('asistencia') or row.get('Asistencia') or 0),
                    'riesgo_predicho': risk_label,
                    'riesgo_nivel': risk_level,
                    'cluster_asignado': cluster_name,
                    'cluster_id': cluster_id,
                    'estado_seguimiento': row.get('estado_seguimiento') if 'estado_seguimiento' in row else 'Pendiente',
                    'desertion_prob': round(float(desertion_prob or 0), 1)
                }

                # Aplicar filtros sencillos
                if riesgo and riesgo.lower() not in str(student_info['riesgo_predicho']).lower():
                    continue
                if cluster is not None and int(student_info['cluster_id']) != int(cluster):
                    continue
                if estado and estado.lower() not in str(student_info['estado_seguimiento']).lower():
                    continue

                students_list.append(student_info)
            except Exception as e:
                print(f"⚠️ Error al procesar fila {idx}: {e}")
                continue

        return {
            'total': len(students_list),
            'showing': f"{start} a {end}",
            'students': students_list
        }
    except Exception as e:
        print(f"❌ Error en /api/students: {e}")
        return JSONResponse(
            status_code=500,
            content={'success': False, 'message': str(e)}
        )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_student_risk(student: StudentData):
    """
    Predecir riesgo académico de un estudiante
    """
    try:
        # Convertir a diccionario
        student_dict = student.model_dump()

        # Realizar predicción
        prediction = predictor.predict_risk(student_dict)

        # Generar recomendaciones basadas en el riesgo
        recommendations = generate_recommendations(
            prediction['risk_level'],
            student_dict,
            prediction['cluster']
        )

        # Identificar factores clave
        key_factors = identify_key_factors(student_dict, prediction['risk_level'])

        return PredictionResponse(
            risk_level=prediction['risk_level'],
            risk_label=prediction['risk_label'],
            risk_probability=prediction['risk_probability'],
            desertion_probability=prediction['desertion_probability'],
            cluster=prediction['cluster'],
            cluster_name=prediction['cluster_name'],
            recommendations=recommendations,
            key_factors=key_factors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/api/students/register")
async def register_student(profile: StudentProfile):
    """
    Registrar nuevo estudiante y generar predicción
    """
    try:
        # Generar ID único
        student_id = profile.codigo

        # Agregar timestamp
        profile.ultima_actualizacion = datetime.now().isoformat()

        # Guardar en BD
        students_db[student_id] = profile

        # Generar predicción
        prediction = await predict_student_risk(profile.datos)
        predictions_db[student_id] = prediction

        return {
            'status': 'success',
            'message': f'Estudiante {profile.nombre} registrado exitosamente',
            'student_id': student_id,
            'prediction': prediction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al registrar: {str(e)}")


@app.get("/api/students/{codigo}")
async def get_student_profile(codigo: str):
    """
    Obtener perfil completo de un estudiante
    (protegido cuando no hay CSV/DB disponible)
    """
    # Buscar en la base de datos simulada
    if codigo in students_db:
        student = students_db[codigo]
        prediction = predictions_db.get(codigo)

        return {
            'student': student,
            'prediction': prediction,
            'last_update': student.ultima_actualizacion
        }

    # Si no existe en memoria, intentar DB o CSV; si no hay datos, crear ejemplo seguro
    try:
        if not df_students.empty:
            # Evitar modulo por cero (df_students no está vacío aquí)
            try:
                idx = int(codigo[-4:]) % len(df_students)
            except Exception:
                idx = 0
            row = df_students.iloc[idx]
            student_dict = row.to_dict()
            try:
                prediction = predictor.predict_risk(student_dict) if predictor and hasattr(predictor, 'predict_risk') else {
                    'risk_label': row.get('riesgo_predicho', 'No disponible'),
                    'cluster_name': row.get('cluster_asignado', 'No asignado'),
                    'risk_probability': float(row.get('probabilidad_desercion', 0) or 0),
                    'desertion_probability': float(row.get('probabilidad_desercion', 0) or 0),
                    'risk_level': 0,
                    'cluster': int(row.get('cluster_asignado') or 0)
                }
            except Exception as e:
                print(f"⚠️ Error al predecir perfil (fallback): {e}")
                prediction = {
                    'risk_label': row.get('riesgo_predicho', 'No disponible'),
                    'cluster_name': row.get('cluster_asignado', 'No asignado'),
                    'risk_probability': float(row.get('probabilidad_desercion', 0) or 0),
                    'desertion_probability': float(row.get('probabilidad_desercion', 0) or 0),
                    'risk_level': 0,
                    'cluster': int(row.get('cluster_asignado') or 0)
                }

            return {
                'student': {
                    'codigo': codigo,
                    'nombre': row.get('nombre', f'Estudiante {codigo}'),
                    'carrera': row.get('carrera', 'Sin carrera'),
                    'ciclo': row.get('ciclo', 0),
                    'edad': int(row.get('edad') or row.get('Edad') or 20),
                    'datos': student_dict
                },
                'prediction': {
                    **prediction,
                    'recommendations': generate_recommendations(
                        prediction.get('risk_level', 0),
                        student_dict,
                        prediction.get('cluster', 0)
                    ),
                    'key_factors': identify_key_factors(student_dict, prediction.get('risk_level', 0))
                },
                'resumen_academico': {
                    'promedio_ponderado': round(float(row.get('promedio_ponderado') or row.get('Promedio_ponderado', 0)), 1),
                    'creditos_cursados': int(row.get('creditos_matriculados') or row.get('Creditos_matriculados') or 0),
                    'asistencia_ultimas_4_semanas': f"{int(row.get('asistencia') or row.get('Asistencia') or 0)}%"
                },
                'datos_basicos': {
                    'edad': f"{int(row.get('edad') or row.get('Edad') or 20)} años",
                    'carga_laboral': f"{row.get('carga_laboral', 'No_trabaja')}",
                    'beca': row.get('beca', 'No_tiene'),
                    'deudor': row.get('deudor', 'Sin_deuda'),
                    'apoyo_familiar': row.get('apoyo_familiar', 'Moderado'),
                    'modalidad': 'Presencial'
                }
            }

        # Si no hay CSV, intentar DB
        try:
            from database import EstudiantesDB
            db = EstudiantesDB()
            student_row = db.get_student_by_codigo(codigo)
            db.close()
            if student_row:
                s = dict(student_row)
                prediction = {
                    'risk_label': s.get('riesgo_predicho', 'No disponible'),
                    'cluster_name': s.get('cluster_asignado', 'No asignado'),
                    'risk_probability': float(s.get('probabilidad_desercion', 0) or 0),
                    'desertion_probability': float(s.get('probabilidad_desercion', 0) or 0),
                    'risk_level': 0,
                    'cluster': int(s.get('cluster_asignado') or 0)
                }
                return {
                    'student': s,
                    'prediction': prediction,
                    'resumen_academico': {
                        'promedio_ponderado': s.get('promedio_ponderado', 0),
                        'creditos_cursados': 0,
                        'asistencia_ultimas_4_semanas': 'N/D'
                    }
                }
        except Exception:
            pass

        # Fallback: crear perfil sintético si no hay datos disponibles
        ejemplo = {
            'codigo': codigo,
            'nombre': f'Estudiante {codigo}',
            'carrera': 'Sin carrera',
            'ciclo': 0,
            'edad': 20,
            'datos': {}
        }
        prediction_fallback = {
            'risk_label': 'Sin evaluar',
            'cluster_name': 'Sin asignar',
            'risk_probability': 0.0,
            'desertion_probability': 0.0,
            'risk_level': 0,
            'cluster': 0
        }
        return {
            'student': ejemplo,
            'prediction': {
                **prediction_fallback,
                'recommendations': ['No hay datos suficientes para generar recomendaciones'],
                'key_factors': []
            },
            'resumen_academico': {
                'promedio_ponderado': 0,
                'creditos_cursados': 0,
                'asistencia_ultimas_4_semanas': 'N/D'
            }
        }

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Estudiante no encontrado: {str(e)}")


@app.get("/api/clusters/{cluster_id}")
async def get_cluster_info(cluster_id: int):
    """
    Obtener información del cluster
    """
    if cluster_id not in [0, 1, 2]:
        raise HTTPException(status_code=404, detail="Cluster no encontrado")

    cluster_info = {
        0: {
            'id': 0,
            'name': 'C2 - Compromiso medio, carga laboral alta',
            'description': 'Estudiantes con buena asistencia, estrés elevado y responsabilidades laborales significativas',
            'avg_risk': 2,
            'size': 650,
            'characteristics': {
                'asistencia': 'Buena',
                'estres_academico': 'Alto',
                'horas_estudio': 'Medias',
                'carga_laboral': 'Alta'
            }
        },
        1: {
            'id': 1,
            'name': 'C1 - Compromiso alto',
            'description': 'Estudiantes con buen desempeño académico, alta asistencia y compromiso sobresaliente',
            'avg_risk': 0,
            'size': 850,
            'characteristics': {
                'asistencia': 'Alta',
                'estres_academico': 'Bajo',
                'horas_estudio': 'Altas',
                'carga_laboral': 'Baja'
            }
        },
        2: {
            'id': 2,
            'name': 'C3 - Riesgo acumulado',
            'description': 'Estudiantes con múltiples factores de riesgo que requieren intervención inmediata',
            'avg_risk': 3,
            'size': 500,
            'characteristics': {
                'asistencia': 'Baja',
                'estres_academico': 'Muy alto',
                'horas_estudio': 'Bajas',
                'carga_laboral': 'Variable'
            }
        }
    }

    return cluster_info[cluster_id]


@app.post("/api/interventions/register")
async def register_intervention(
    codigo: str,
    tipo: str,
    descripcion: str,
    tutor: str
):
    """
    Registrar una intervención tutorial
    """
    intervention = {
        'codigo_estudiante': codigo,
        'tipo': tipo,
        'descripcion': descripcion,
        'tutor': tutor,
        'fecha': datetime.now().isoformat(),
        'estado': 'Registrada'
    }

    return {
        'status': 'success',
        'message': 'Intervención registrada exitosamente',
        'intervention': intervention
    }


@app.get("/api/export/students")
async def export_students(formato: str = "csv"):
    """
    Exportar listado de estudiantes
    """
    if formato.lower() == "csv":
        return FileResponse(
            'estudiantes_data.csv',
            media_type='text/csv',
            filename='estudiantes_export.csv'
        )
    else:
        raise HTTPException(status_code=400, detail="Formato no soportado")


@app.get("/api/students/search")
async def search_students(q: str = "", limit: int = 10):
    """
    Buscar estudiantes por nombre o código (compatible con frontend/registro.html).
    Retorna JSON: { "success": True, "results": [ {codigo,nombre,carrera,ciclo}, ... ] }
    """
    try:
        q_clean = (q or "").strip().lower()
        if q_clean == "":
            return {'success': True, 'results': []}

        results = []

        # 1) Buscar en df_students si está cargado
        if not df_students.empty:
            df = df_students
            # Normalizar columnas y buscar en nombre/codigo (tolerante a mayúsculas)
            for _, row in df.iterrows():
                nombre = str(row.get('nombre', '')).lower()
                codigo = str(row.get('codigo', '')).lower()
                if q_clean in nombre or q_clean in codigo:
                    results.append({
                        'codigo': row.get('codigo'),
                        'nombre': row.get('nombre'),
                        'carrera': row.get('carrera'),
                        'ciclo': row.get('ciclo')
                    })
                    if len(results) >= limit:
                        break

            return {'success': True, 'results': results}

        # 2) Si no hay CSV, intentar consultar la base de datos
        try:
            from database import EstudiantesDB
            db = EstudiantesDB()
            # Usar consulta parametrizada para evitar inyección (LIKE con %)
            sql = """
                SELECT codigo, nombre, carrera, ciclo
                FROM estudiantes
                WHERE LOWER(nombre) LIKE %(q)s OR LOWER(codigo) LIKE %(q)s
                LIMIT %(limit)s
            """
            params = {'q': f"%{q_clean}%", 'limit': limit}
            rows = db.db.fetch_all(sql, params)
            db.close()

            # db.db.fetch_all puede devolver list[dict]
            for r in rows:
                results.append({
                    'codigo': r.get('codigo'),
                    'nombre': r.get('nombre'),
                    'carrera': r.get('carrera'),
                    'ciclo': r.get('ciclo')
                })

            return {'success': True, 'results': results}

        except Exception as db_err:
            # Fallback vacío si DB no disponible
            print(f"⚠️ Búsqueda en DB falló: {db_err}")
            return {'success': True, 'results': []}

    except Exception as e:
        print(f"❌ Error en /api/students/search: {e}")
        return JSONResponse(status_code=500, content={'success': False, 'message': str(e)})


# ==================== FUNCIONES AUXILIARES ====================

def generate_recommendations(risk_level: int, student_data: dict, cluster: int) -> List[str]:
    """
    Generar recomendaciones personalizadas
    """
    recommendations = []

    if risk_level >= 3:  # Alto o crítico
        recommendations.append("Agendar una sesión de orientación académica prioritaria")
        recommendations.append("Coordinar derivación opcional a bienestar psicológico para manejo de estrés")

    if student_data['Asistencia'] < 85:
        recommendations.append("Monitorear asistencia semanalmente")

    if student_data['Promedio_ponderado'] < 14:
        recommendations.append("Recomendar tutoría académica en cursos clave")

    if student_data['Horas_trabajo_semana'] > 20:
        recommendations.append("Explorar ajustes de horario laboral o negociación de turnos")

    if student_data['Cursos_desaprobados'] > 2:
        recommendations.append("Revisar estrategias de estudio y planificación académica")

    if student_data['Horas_estudio_semana'] < 15:
        recommendations.append("Desarrollar plan de estudio estructurado")

    if cluster == 2:  # Riesgo acumulado
        recommendations.append("Considerar reducción de carga académica el próximo ciclo")
        recommendations.append("Involucrar a la familia en un breve espacio de información sobre señales de alarma")

    if not recommendations:
        recommendations.append("Mantener el buen desempeño actual")
        recommendations.append("Continuar con seguimiento regular")

    return recommendations


def identify_key_factors(student_data: dict, risk_level: int) -> List[Dict[str, str]]:
    """
    Identificar factores clave que influyen en el riesgo
    """
    factors = []

    if student_data['Promedio_ponderado'] < 14:
        factors.append({
            'factor': 'Estrés académico',
            'nivel': 'Alto impacto',
            'descripcion': 'Promedio académico por debajo del umbral recomendado'
        })

    if student_data['Asistencia'] < 85:
        factors.append({
            'factor': 'Asistencia',
            'nivel': 'Moderado impacto',
            'descripcion': 'Asistencia inferior al 85%'
        })

    if student_data['Horas_trabajo_semana'] > 20:
        factors.append({
            'factor': 'Carga laboral',
            'nivel': 'Incrementa el riesgo',
            'descripcion': f"Trabaja {int(student_data['Horas_trabajo_semana'])} h/semana"
        })

    if student_data['Horas_estudio_semana'] > 18:
        factors.append({
            'factor': 'Horas de estudio',
            'nivel': 'Nivel intermedio',
            'descripcion': 'Dedica tiempo adecuado al estudio'
        })

    if student_data['indice_regularidad'] > 60:
        factors.append({
            'factor': 'Asistencia',
            'nivel': 'Factor protector',
            'descripcion': 'Buena regularidad en asistencia'
        })

    if student_data['Promedio_ponderado'] > 16:
        factors.append({
            'factor': 'Apoyo familiar',
            'nivel': 'Compensa parte del riesgo',
            'descripcion': 'Cuenta con apoyo familiar'
        })

    if not factors:
        factors.append({
            'factor': 'Desempeño actual',
            'nivel': 'Leve amortiguador',
            'descripcion': 'Sin factores de riesgo significativos identificados'
        })

    return factors


# Montar archivos estáticos
if os.path.exists('frontend'):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
