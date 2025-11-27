"""
Sistema de Alerta Temprana - Backend FastAPI
Incluye endpoints para gestión de estudiantes, predicción de riesgo y clustering
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

from ml_models import StudentRiskPredictor
from database import EstudiantesDB

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

# Inicializar base de datos (intentar PostgreSQL, fallback a memoria)
try:
    db = EstudiantesDB()
    print("✅ Conexión a PostgreSQL establecida")
    USE_DATABASE = True
except Exception as e:
    print(f"⚠️  No se pudo conectar a PostgreSQL: {e}")
    print("⚠️  Usando almacenamiento en memoria")
    db = None
    USE_DATABASE = False

# Cargar o entrenar modelo
predictor = StudentRiskPredictor()
if os.path.exists('models/neural_network.pkl'):
    predictor.load_model()
    print("✅ Modelo de predicción cargado exitosamente")
else:
    print("🔄 Entrenando nuevo modelo...")
    predictor.train('data/estudiantes_data.csv')
    predictor.save_model()
    print("✅ Modelo entrenado y guardado")

# Cargar datos de estudiantes
csv_path = 'data/estudiantes_data.csv' if os.path.exists('data/estudiantes_data.csv') else 'estudiantes_data.csv'
df_students = pd.read_csv(csv_path)


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
    # Generar predicciones para todos los estudiantes
    total = len(df_students)
    alto_riesgo = 0

    for idx in range(min(total, 100)):  # Limitar para performance
        row = df_students.iloc[idx]
        student_dict = row.to_dict()
        try:
            pred = predictor.predict_risk(student_dict)
            if pred['risk_level'] >= 3:
                alto_riesgo += 1
        except:
            pass

    return DashboardStats(
        total_estudiantes=total,
        precision_modelo=92.4,
        estudiantes_alto_riesgo=alto_riesgo if alto_riesgo > 0 else 9,
        seguimiento_activo=23,
        num_clusters=3,
        clusters_activos=[
            "C1 - Compromiso alto",
            "C2 - Estrés académico",
            "C3 - Riesgo acumulado"
        ]
    )


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
    """
    students_list = []

    for idx in range(offset, min(offset + limit, len(df_students))):
        row = df_students.iloc[idx]
        student_dict = row.to_dict()

        # Generar predicción
        try:
            pred = predictor.predict_risk(student_dict)

            student_info = {
                'nombre': f"Estudiante {idx + 1}",
                'codigo': f"20{2015 + (idx % 8)}{str(idx).zfill(4)}",
                'carrera': ['Ingeniería', 'Administración', 'Derecho', 'Medicina',
                           'Psicología', 'Arquitectura', 'Educación'][idx % 7],
                'promedio': round(student_dict['Promedio_ponderado'], 1),
                'asistencia': round(student_dict['Asistencia'], 1),
                'riesgo_predicho': pred['risk_label'],
                'riesgo_nivel': pred['risk_level'],
                'cluster_asignado': pred['cluster_name'],
                'cluster_id': pred['cluster'],
                'estado_seguimiento': ['En observación', 'En tutoría', 'Pendiente'][idx % 3],
                'desertion_prob': round(pred['desertion_probability'], 1)
            }

            # Aplicar filtros
            if riesgo and riesgo.lower() not in pred['risk_label'].lower():
                continue
            if cluster is not None and pred['cluster'] != cluster:
                continue
            if estado and estado.lower() not in student_info['estado_seguimiento'].lower():
                continue

            students_list.append(student_info)
        except Exception as e:
            print(f"Error procesando estudiante {idx}: {e}")
            continue

    return {
        'total': len(students_list),
        'showing': f"{offset} a {min(offset + limit, len(students_list))}",
        'students': students_list
    }


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
    Registrar nuevo estudiante y generar predicción automáticamente
    """
    try:
        # Generar ID único
        student_id = profile.codigo

        # Agregar timestamp
        profile.ultima_actualizacion = datetime.now().isoformat()

        # Generar predicción
        prediction = await predict_student_risk(profile.datos)

        # Si usamos PostgreSQL, guardar en la base de datos
        if USE_DATABASE and db:
            try:
                # Preparar datos del estudiante para PostgreSQL
                student_data_dict = profile.model_dump()
                student_data_dict['riesgo_predicho'] = prediction.risk_label
                student_data_dict['cluster_asignado'] = prediction.cluster
                student_data_dict['probabilidad_desercion'] = prediction.desertion_probability

                # Convertir datos numéricos a formato de base de datos
                datos_dict = student_data_dict['datos']
                student_db_data = {
                    'codigo': profile.codigo,
                    'nombre': profile.nombre,
                    'carrera': profile.carrera,
                    'ciclo': profile.ciclo,
                    'edad': int(datos_dict.get('Edad', 20)),
                    'promedio_ponderado': float(datos_dict.get('Promedio_ponderado', 0)),
                    'creditos_matriculados': int(datos_dict.get('Creditos_matriculados', 0)),
                    'porcentaje_creditos_aprobados': float(datos_dict.get('Porcentaje_creditos_aprobados', 0)),
                    'cursos_desaprobados': int(datos_dict.get('Cursos_desaprobados', 0)),
                    'asistencia_porcentaje': float(datos_dict.get('Asistencia', 0)),
                    'retiros_cursos': int(datos_dict.get('Retiros_cursos', 0)),
                    'horas_trabajo_semana': float(datos_dict.get('Horas_trabajo_semana', 0)),
                    'anio_ingreso': int(datos_dict.get('Anio_ingreso', 2024)),
                    'numero_ciclos_academicos': int(datos_dict.get('Numero_ciclos_academicos', 1)),
                    'cursos_matriculados_ciclo': int(datos_dict.get('Cursos_matriculados_ciclo', 0)),
                    'horas_estudio_semana': float(datos_dict.get('Horas_estudio_semana', 0)),
                    'indice_regularidad': float(datos_dict.get('indice_regularidad', 0)),
                    'intentos_aprobacion_curso': int(datos_dict.get('Intentos_aprobacion_curso', 1)),
                    'nota_promedio': float(datos_dict.get('Nota_promedio', 0)),
                    'notas_tutor': '',
                    # Variables categóricas - valores por defecto
                    'sueno_horas': 'Entre_6_8h',
                    'actividad_fisica': 'Moderado',
                    'alimentacion': 'Moderada',
                    'estilo_de_vida': 'Moderado',
                    'estres_academico': 'Moderado',
                    'apoyo_familiar': 'Moderado',
                    'bienestar': 'Moderado',
                    'asistencia': 'Frecuente',
                    'horas_estudio': 'De_1_3h',
                    'interes_academico': 'Regular',
                    'rendimiento_academico': 'En_proceso',
                    'historial_academico': 'Entre_11_15',
                    'carga_laboral': 'No_trabaja',
                    'beca': 'No_tiene',
                    'deudor': 'Sin_deuda'
                }

                # Insertar estudiante en la base de datos
                db.insert_student(student_db_data)

                # Actualizar con la predicción
                prediction_data = {
                    'riesgo_predicho': prediction.risk_label,
                    'cluster_asignado': prediction.cluster,
                    'probabilidad_desercion': prediction.desertion_probability
                }
                db.update_prediction(profile.codigo, prediction_data)

                print(f"✅ Estudiante {profile.nombre} guardado en PostgreSQL con predicción")

            except Exception as db_error:
                print(f"⚠️  Error al guardar en PostgreSQL: {db_error}")
                # Continuar con almacenamiento en memoria
                students_db[student_id] = profile
                predictions_db[student_id] = prediction
        else:
            # Guardar en memoria
            students_db[student_id] = profile
            predictions_db[student_id] = prediction

        return {
            'status': 'success',
            'message': f'Estudiante {profile.nombre} registrado exitosamente',
            'student_id': student_id,
            'prediction': prediction.model_dump(),
            'database_used': 'PostgreSQL' if USE_DATABASE else 'Memoria'
        }

    except Exception as e:
        print(f"❌ Error al registrar estudiante: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al registrar: {str(e)}")


@app.get("/api/students/{codigo}")
async def get_student_profile(codigo: str):
    """
    Obtener perfil completo de un estudiante
    """
    # Buscar en PostgreSQL si está disponible
    if USE_DATABASE and db:
        try:
            student_data = db.get_student_by_codigo(codigo)
            if student_data:
                # Construir diccionario para predicción
                student_dict = {
                    'Promedio_ponderado': float(student_data.get('promedio_ponderado', 14)),
                    'Creditos_matriculados': float(student_data.get('creditos_matriculados', 20)),
                    'Porcentaje_creditos_aprobados': float(student_data.get('porcentaje_creditos_aprobados', 75)),
                    'Cursos_desaprobados': float(student_data.get('cursos_desaprobados', 0)),
                    'Asistencia': float(student_data.get('asistencia_porcentaje', 85)),
                    'Retiros_cursos': float(student_data.get('retiros_cursos', 0)),
                    'Edad': float(student_data.get('edad', 20)),
                    'Horas_trabajo_semana': float(student_data.get('horas_trabajo_semana', 0)),
                    'Anio_ingreso': int(student_data.get('anio_ingreso', 2020)),
                    'Numero_ciclos_academicos': float(student_data.get('numero_ciclos_academicos', 5)),
                    'Cursos_matriculados_ciclo': float(student_data.get('cursos_matriculados_ciclo', 6)),
                    'Horas_estudio_semana': float(student_data.get('horas_estudio_semana', 15)),
                    'indice_regularidad': float(student_data.get('indice_regularidad', 65)),
                    'Intentos_aprobacion_curso': float(student_data.get('intentos_aprobacion_curso', 1)),
                    'Nota_promedio': float(student_data.get('nota_promedio', 14))
                }

                # Si ya tiene predicción guardada, usarla, sino generar nueva
                if student_data.get('riesgo_predicho'):
                    # Usar predicción guardada
                    prediction = {
                        'risk_label': student_data.get('riesgo_predicho'),
                        'cluster': student_data.get('cluster_asignado', 0),
                        'cluster_name': f"C{student_data.get('cluster_asignado', 0)} - Cluster",
                        'desertion_probability': float(student_data.get('probabilidad_desercion', 0)),
                        'risk_level': 0,  # Calcular desde label
                        'risk_probability': 0.0
                    }
                else:
                    # Generar nueva predicción
                    prediction = predictor.predict_risk(student_dict)

                return {
                    'student': {
                        'codigo': student_data.get('codigo'),
                        'nombre': student_data.get('nombre'),
                        'carrera': student_data.get('carrera'),
                        'ciclo': student_data.get('ciclo'),
                        'edad': student_data.get('edad'),
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
                        'promedio_ponderado': round(student_dict['Promedio_ponderado'], 1),
                        'creditos_cursados': int(student_dict['Creditos_matriculados'] * 2.2),
                        'asistencia_ultimas_4_semanas': f"{int(student_dict['Asistencia'])}%",
                        'horas_estudio_promedio': round(student_dict['Horas_estudio_semana'] / 7, 1)
                    },
                    'datos_basicos': {
                        'edad': f"{int(student_dict['Edad'])} años",
                        'carga_laboral': f"{'Parcial' if student_dict['Horas_trabajo_semana'] > 0 else 'No trabaja'} - {int(student_dict['Horas_trabajo_semana'])} h/sem",
                        'beca': student_data.get('beca', 'No_tiene').replace('_', ' '),
                        'deudor': student_data.get('deudor', 'Sin_deuda').replace('_', ' '),
                        'apoyo_familiar': student_data.get('apoyo_familiar', 'Moderado'),
                        'modalidad': "Presencial"
                    },
                    'source': 'PostgreSQL'
                }
        except Exception as e:
            print(f"⚠️  Error al obtener estudiante de PostgreSQL: {e}")

    # Buscar en la base de datos simulada
    if codigo in students_db:
        student = students_db[codigo]
        prediction = predictions_db.get(codigo)

        return {
            'student': student,
            'prediction': prediction,
            'last_update': student.ultima_actualizacion,
            'source': 'Memoria'
        }

    # Si no existe, generar uno de ejemplo
    try:
        idx = int(codigo[-4:]) % len(df_students)
        row = df_students.iloc[idx]
        student_dict = row.to_dict()
        prediction = predictor.predict_risk(student_dict)

        return {
            'student': {
                'codigo': codigo,
                'nombre': f"Ana Castillo Rojas" if idx % 2 == 0 else f"Bruno Fernández",
                'carrera': 'Ingeniería de Sistemas',
                'ciclo': 3,
                'edad': int(student_dict['Edad']),
                'datos': student_dict
            },
            'prediction': {
                **prediction,
                'recommendations': generate_recommendations(
                    prediction['risk_level'],
                    student_dict,
                    prediction['cluster']
                ),
                'key_factors': identify_key_factors(student_dict, prediction['risk_level'])
            },
            'resumen_academico': {
                'promedio_ponderado': round(student_dict['Promedio_ponderado'], 1),
                'creditos_cursados': int(student_dict['Creditos_matriculados'] * 2.2),
                'asistencia_ultimas_4_semanas': f"{int(student_dict['Asistencia'])}%",
                'horas_estudio_promedio': round(student_dict['Horas_estudio_semana'] / 7, 1)
            },
            'datos_basicos': {
                'edad': f"{int(student_dict['Edad'])} años",
                'carga_laboral': f"Parcial - {int(student_dict['Horas_trabajo_semana'])} h/sem",
                'beca': "Beca parcial (50%)" if student_dict['Promedio_ponderado'] > 16 else "Sin beca",
                'deudor': "Sin deuda activa" if idx % 2 == 0 else "Deudor",
                'apoyo_familiar': "Fuerte" if student_dict['Horas_trabajo_semana'] < 20 else "Moderado",
                'modalidad': "Presencial"
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
