"""
Sistema de Alerta Temprana - Backend FastAPI
Backend completo con PostgreSQL para registro, panel y perfil de estudiantes
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import json
from datetime import datetime
import os

from database import EstudiantesDB, EntrenamientosDB

# Inicializar FastAPI
app = FastAPI(
    title="Sistema de Alerta Temprana",
    description="API para predicción de riesgo académico y clustering de estudiantes",
    version="2.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELOS PYDANTIC ====================

class EstudianteRegistro(BaseModel):
    """Modelo para registro de estudiante"""
    codigo: str
    nombre: str
    carrera: str
    ciclo: int
    edad: int = Field(default=20)
    promedio_ponderado: float = Field(default=0.0)

    # Variables categóricas (15 variables del modelo)
    sueno_horas: Optional[str] = None
    actividad_fisica: Optional[str] = None
    alimentacion: Optional[str] = None
    estilo_de_vida: Optional[str] = None
    estres_academico: Optional[str] = None
    apoyo_familiar: Optional[str] = None
    bienestar: Optional[str] = None
    asistencia: Optional[str] = None
    horas_estudio: Optional[str] = None
    interes_academico: Optional[str] = None
    rendimiento_academico: Optional[str] = None
    historial_academico: Optional[str] = None
    carga_laboral: Optional[str] = None
    beca: Optional[str] = None
    deudor: Optional[str] = None

    notas_tutor: Optional[str] = None


# ==================== ENDPOINTS HTML ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Página principal - Formulario de registro"""
    try:
        with open('frontend/registro.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: No se encontró el archivo registro.html</h1>",
            status_code=404
        )


@app.get("/registro", response_class=HTMLResponse)
async def registro():
    """Página de registro - Formulario de registro"""
    try:
        with open('frontend/registro.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: No se encontró el archivo registro.html</h1>",
            status_code=404
        )


@app.get("/panel", response_class=HTMLResponse)
async def panel_tutor():
    """Panel del tutor"""
    try:
        with open('frontend/panel.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: No se encontró el archivo panel.html</h1>",
            status_code=404
        )


@app.get("/comparacion", response_class=HTMLResponse)
async def comparacion_modelos():
    """Página de comparación de modelos"""
    try:
        with open('frontend/comparacion.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: No se encontró el archivo comparacion.html</h1>",
            status_code=404
        )


@app.get("/perfil/{codigo}", response_class=HTMLResponse)
async def perfil_estudiante(codigo: str):
    """Perfil del estudiante"""
    try:
        with open('frontend/perfil.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: No se encontró el archivo perfil.html</h1>",
            status_code=404
        )


@app.get("/perfil", response_class=HTMLResponse)
async def perfil_estudiante_query():
    """Servir la página de perfil (soporta URL con ?codigo=... desde el frontend antiguo)"""
    try:
        with open('frontend/perfil.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: No se encontró el archivo perfil.html</h1>",
            status_code=404
        )


# ==================== ENDPOINTS API ====================

@app.get("/api/stats")
async def get_dashboard_stats():
    """Obtener estadísticas del dashboard desde PostgreSQL"""
    try:
        db = EstudiantesDB()
        stats = db.get_statistics()

        # Obtener información del último entrenamiento
        entrenamientos_db = EntrenamientosDB()
        ultimo_entrenamiento = entrenamientos_db.get_ultimo_entrenamiento()
        entrenamientos_db.close()

        precision_modelo = 92.4
        if ultimo_entrenamiento and ultimo_entrenamiento.get('precision_modelo'):
            precision_modelo = float(ultimo_entrenamiento['precision_modelo'])

        db.close()

        return {
            "total_estudiantes": stats.get('total_estudiantes', 0),
            "precision_modelo": precision_modelo,
            "estudiantes_alto_riesgo": stats.get('estudiantes_alto_riesgo', 0),
            "seguimiento_activo": max(0, stats.get('total_estudiantes', 0) // 5),
            "num_clusters": 3,
            "clusters_activos": [
                "C1 - Compromiso alto",
                "C2 - Estrés académico",
                "C3 - Riesgo acumulado"
            ]
        }
    except Exception as e:
        print(f"❌ Error en /api/stats: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/students")
async def get_all_students(
    riesgo: Optional[str] = None,
    cluster: Optional[int] = None,
    estado: Optional[str] = None,
    busqueda: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
):
    """Obtener lista de estudiantes desde PostgreSQL con filtros"""
    try:
        db = EstudiantesDB()
        df = db.get_all_students()
        db.close()

        if df.empty:
            return {
                'total': 0,
                'showing': '0 a 0',
                'students': []
            }

        students_list = []

        for idx, row in df.iterrows():
            student_dict = row.to_dict()

            # Aplicar filtros
            if riesgo and riesgo.lower() != 'todos':
                if not student_dict.get('riesgo_predicho'):
                    continue
                if riesgo.lower() not in str(student_dict['riesgo_predicho']).lower():
                    continue

            if cluster is not None and cluster != -1:
                if student_dict.get('cluster_asignado') != cluster:
                    continue

            if estado and estado.lower() != 'todos':
                if not student_dict.get('estado_seguimiento'):
                    continue
                if estado.lower() not in str(student_dict['estado_seguimiento']).lower():
                    continue

            # Filtro de búsqueda por nombre o código
            if busqueda:
                busqueda_lower = busqueda.lower()
                nombre = str(student_dict.get('nombre', '')).lower()
                codigo = str(student_dict.get('codigo', '')).lower()
                if busqueda_lower not in nombre and busqueda_lower not in codigo:
                    continue

            student_info = {
                'nombre': student_dict.get('nombre', 'Sin nombre'),
                'codigo': student_dict.get('codigo', 'N/A'),
                'carrera': student_dict.get('carrera', 'Sin carrera'),
                'promedio': round(float(student_dict.get('promedio_ponderado', 0) or 0), 1),
                'riesgo_predicho': student_dict.get('riesgo_predicho', 'Sin evaluar'),
                'cluster_asignado': get_cluster_name(student_dict.get('cluster_asignado')),
                'estado_seguimiento': student_dict.get('estado_seguimiento', 'Pendiente'),
                'probabilidad_desercion': round(float(student_dict.get('probabilidad_desercion', 0) or 0), 1)
            }

            students_list.append(student_info)

        # Aplicar paginación
        total = len(students_list)
        students_paginated = students_list[offset:offset+limit]

        return {
            'total': total,
            'showing': f"{offset} a {min(offset+limit, total)}",
            'students': students_paginated
        }

    except Exception as e:
        print(f"❌ Error en /api/students: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={'success': False, 'message': str(e)}
        )


@app.post("/api/students/register")
async def register_student(data: dict):
    """Registrar nuevo estudiante en PostgreSQL y generar predicción"""
    try:
        print(f"📥 Recibiendo datos para registro: {data}")

        # Extraer datos básicos
        codigo = data.get('codigo')
        nombre = data.get('nombre')
        carrera = data.get('carrera')
        ciclo = data.get('ciclo')
        edad = data.get('edad', 20)
        promedio_ponderado = data.get('promedio_ponderado', 14.0)
        notas_tutor = data.get('notas_tutor')

        # Extraer datos categóricos
        datos_categoricos = data.get('datos', {})

        if not codigo or not nombre:
            raise HTTPException(status_code=400, detail="Código y nombre son requeridos")

        # Preparar datos para inserción
        student_data = {
            'codigo': codigo,
            'nombre': nombre,
            'carrera': carrera,
            'ciclo': ciclo,
            'edad': edad,
            'promedio_ponderado': promedio_ponderado,
            'notas_tutor': notas_tutor,

            # Variables categóricas
            'sueno_horas': datos_categoricos.get('sueno_horas'),
            'actividad_fisica': datos_categoricos.get('actividad_fisica'),
            'alimentacion': datos_categoricos.get('alimentacion'),
            'estilo_de_vida': datos_categoricos.get('estilo_de_vida'),
            'estres_academico': datos_categoricos.get('estres_academico'),
            'apoyo_familiar': datos_categoricos.get('apoyo_familiar'),
            'bienestar': datos_categoricos.get('bienestar'),
            'asistencia': datos_categoricos.get('asistencia'),
            'horas_estudio': datos_categoricos.get('horas_estudio'),
            'interes_academico': datos_categoricos.get('interes_academico'),
            'rendimiento_academico': datos_categoricos.get('rendimiento_academico'),
            'historial_academico': datos_categoricos.get('historial_academico'),
            'carga_laboral': datos_categoricos.get('carga_laboral'),
            'beca': datos_categoricos.get('beca'),
            'deudor': datos_categoricos.get('deudor'),
        }

        # Insertar en base de datos
        db = EstudiantesDB()
        success = db.insert_student(student_data)

        if not success:
            db.close()
            raise HTTPException(status_code=500, detail="Error al insertar estudiante en la BD")

        # Generar predicción (usaremos el modelo bayesiano aquí)
        prediction = generar_prediccion_bayesiana(datos_categoricos)

        # Actualizar con la predicción
        prediction_data = {
            'riesgo_predicho': prediction['risk_label'],
            'cluster_asignado': prediction['cluster'],
            'probabilidad_desercion': prediction['desertion_probability']
        }

        db.update_prediction(codigo, prediction_data)
        db.close()

        print(f"✅ Estudiante {nombre} registrado exitosamente con predicción")

        return {
            'status': 'success',
            'message': f'Estudiante {nombre} registrado exitosamente',
            'student_id': codigo,
            'prediction': prediction
        }

    except Exception as e:
        print(f"❌ Error al registrar estudiante: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/students/search")
async def search_students(q: str = "", limit: int = 10):
    """Buscar estudiantes por nombre o código en PostgreSQL"""
    try:
        if not q or len(q) < 2:
            return {'success': True, 'results': []}

        db = EstudiantesDB()

        # Buscar en la base de datos
        query = """
        SELECT codigo, nombre, carrera, ciclo
        FROM estudiantes
        WHERE LOWER(nombre) LIKE %s OR LOWER(codigo) LIKE %s
        LIMIT %s
        """
        q_pattern = f"%{q.lower()}%"
        results = db.db.fetch_all(query, (q_pattern, q_pattern, limit))
        db.close()

        return {
            'success': True,
            'results': [dict(r) for r in results]
        }

    except Exception as e:
        print(f"❌ Error en búsqueda: {e}")
        return JSONResponse(
            status_code=500,
            content={'success': False, 'message': str(e)}
        )


@app.get("/api/students/{codigo}")
async def get_student_profile(codigo: str):
    """Obtener perfil completo de un estudiante desde PostgreSQL (con fallback a CSV)"""
    try:
        db = EstudiantesDB()
        student = db.get_student_by_codigo(codigo)
        db.close()

        if student:
            student_dict = dict(student)
            return {
                'student': {
                    'codigo': student_dict.get('codigo'),
                    'nombre': student_dict.get('nombre'),
                    'carrera': student_dict.get('carrera'),
                    'ciclo': student_dict.get('ciclo'),
                    'edad': student_dict.get('edad', 20)
                },
                'prediction': {
                    'risk_label': student_dict.get('riesgo_predicho', 'Sin evaluar'),
                    'cluster_name': get_cluster_name(student_dict.get('cluster_asignado')),
                    'risk_probability': float(student_dict.get('probabilidad_desercion', 0) or 0),
                    'desertion_probability': float(student_dict.get('probabilidad_desercion', 0) or 0),
                    'risk_level': map_risk_to_level(student_dict.get('riesgo_predicho')),
                    'cluster': student_dict.get('cluster_asignado', 0),
                    'recommendations': generar_recomendaciones(student_dict),
                    'key_factors': generar_factores_clave(student_dict)
                },
                'resumen_academico': {
                    'promedio_ponderado': round(float(student_dict.get('promedio_ponderado', 0) or 0), 1),
                    'creditos_cursados': student_dict.get('creditos_matriculados', 0) or 44,
                    'asistencia_ultimas_4_semanas': f"{student_dict.get('asistencia_porcentaje', 87)}%"
                },
                'datos_basicos': {
                    'edad': f"{student_dict.get('edad', 20)} años",
                    'carga_laboral': format_carga_laboral(student_dict.get('carga_laboral')),
                    'beca': format_beca(student_dict.get('beca')),
                    'deudor': format_deudor(student_dict.get('deudor')),
                    'apoyo_familiar': student_dict.get('apoyo_familiar', 'Moderado'),
                    'modalidad': 'Presencial'
                }
            }

        # FALLBACK a CSV local si no está en BD
        possible_paths = [
            'estudiantes_data.csv',
            os.path.join('data', 'estudiantes_data.csv'),
            os.path.join(os.path.dirname(__file__), 'estudiantes_data.csv'),
            os.path.join(os.path.dirname(__file__), 'data', 'estudiantes_data.csv'),
        ]
        for p in possible_paths:
            try:
                if os.path.exists(p):
                    df_csv = pd.read_csv(p, dtype=str).fillna('')
                    matched = df_csv[df_csv.apply(lambda r: str(r.get('codigo','')).strip() == str(codigo).strip(), axis=1)]
                    if matched.shape[0] > 0:
                        row = matched.iloc[0].to_dict()
                        # construir estructuras mínimas esperadas por el frontend
                        student_basic = {
                            'codigo': row.get('codigo', codigo),
                            'nombre': row.get('nombre', row.get('Nombre', 'Sin nombre')),
                            'carrera': row.get('carrera', row.get('Carrera', None)),
                            'ciclo': int(row.get('ciclo', row.get('Ciclo', 0))) if str(row.get('ciclo','')).isdigit() else None,
                            'edad': int(row.get('Edad', 20)) if str(row.get('Edad','')).isdigit() else 20
                        }
                        prediction = {
                            'risk_label': row.get('riesgo_predicho', 'Sin evaluar'),
                            'cluster_name': row.get('cluster_asignado', 'Sin asignar'),
                            'risk_probability': float(row.get('probabilidad_desercion', 0) or 0),
                            'desertion_probability': float(row.get('probabilidad_desercion', 0) or 0),
                            'risk_level': map_risk_to_level(row.get('riesgo_predicho'))
                        }
                        resumen = {
                            'promedio_ponderado': float(row.get('Promedio_ponderado', row.get('promedio_ponderado', 0)) or 0),
                            'creditos_cursados': int(row.get('Creditos_matriculados', 44) or 44),
                            'asistencia_ultimas_4_semanas': f"{int(float(row.get('Asistencia', row.get('asistencia_porcentaje', 87)) or 0))}%"
                        }
                        datos_basicos = {
                            'edad': f"{student_basic.get('edad', 20)} años",
                            'carga_laboral': row.get('carga_laboral', 'No_trabaja'),
                            'beca': row.get('beca', 'No_tiene'),
                            'deudor': row.get('deudor', 'Sin_deuda'),
                            'apoyo_familiar': row.get('apoyo_familiar', 'Moderado'),
                            'modalidad': 'Presencial'
                        }

                        return {
                            'student': student_basic,
                            'prediction': prediction,
                            'resumen_academico': resumen,
                            'datos_basicos': datos_basicos
                        }
            except Exception as e:
                print(f"⚠️ Error leyendo CSV fallback {p}: {e}")
                continue

        # No encontrado en BD ni CSV
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")

    except Exception as e:
        print(f"❌ Error al obtener perfil: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/comparacion-modelos")
async def get_comparacion_modelos(limit: int = 10):
    """Obtener comparación de entrenamientos de modelos con mayor precisión destacada"""
    try:
        entrenamientos_db = EntrenamientosDB()
        historial = entrenamientos_db.get_historial_entrenamientos(limit=limit)
        entrenamientos_db.close()

        if not historial:
            return {
                "total": 0,
                "entrenamientos": [],
                "mejor_modelo": None,
                "estadisticas": {
                    "precision_promedio": 0,
                    "precision_max": 0,
                    "precision_min": 0
                }
            }

        # Procesar datos
        entrenamientos_procesados = []
        precisiones = []

        for e in historial:
            precision = float(e.get('precision_modelo', 0))
            precisiones.append(precision)

            # Parsear métricas JSON
            metricas = {}
            if e.get('metricas_json'):
                try:
                    import json
                    if isinstance(e['metricas_json'], str):
                        metricas = json.loads(e['metricas_json'])
                    else:
                        metricas = e['metricas_json']
                except:
                    metricas = {}

            entrenamientos_procesados.append({
                "id": e.get('id'),
                "fecha_entrenamiento": str(e.get('fecha_entrenamiento')),
                "num_estudiantes": e.get('num_estudiantes_entrenamiento', 0),
                "precision_modelo": precision,
                "modelo_version": e.get('modelo_version', 'v2'),
                "observaciones": e.get('observaciones', ''),
                "metricas": {
                    "train_accuracy": metricas.get('train_accuracy', 0),
                    "test_accuracy": metricas.get('test_accuracy', 0),
                    "n_components": metricas.get('n_components', 0),
                    "gap": abs(metricas.get('train_accuracy', 0) - metricas.get('test_accuracy', 0)) if metricas.get('train_accuracy') and metricas.get('test_accuracy') else 0
                }
            })

        # Encontrar mejor modelo (mayor precisión)
        mejor_indice = precisiones.index(max(precisiones)) if precisiones else 0
        mejor_modelo = entrenamientos_procesados[mejor_indice] if entrenamientos_procesados else None

        # Calcular estadísticas
        estadisticas = {
            "precision_promedio": sum(precisiones) / len(precisiones) if precisiones else 0,
            "precision_max": max(precisiones) if precisiones else 0,
            "precision_min": min(precisiones) if precisiones else 0,
            "total_entrenamientos": len(entrenamientos_procesados)
        }

        return {
            "total": len(entrenamientos_procesados),
            "entrenamientos": entrenamientos_procesados,
            "mejor_modelo": mejor_modelo,
            "estadisticas": estadisticas
        }

    except Exception as e:
        print(f"❌ Error en /api/comparacion-modelos: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/students")
async def export_students(formato: str = "csv"):
    """Exportar listado de estudiantes desde PostgreSQL"""
    try:
        db = EstudiantesDB()
        df = db.get_all_students()
        db.close()

        if formato.lower() == "csv":
            # Guardar temporalmente
            temp_file = "/tmp/estudiantes_export.csv"
            df.to_csv(temp_file, index=False)
            return FileResponse(
                temp_file,
                media_type='text/csv',
                filename=f'estudiantes_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
        else:
            raise HTTPException(status_code=400, detail="Formato no soportado")

    except Exception as e:
        print(f"❌ Error al exportar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== FUNCIONES AUXILIARES ====================

def get_cluster_name(cluster_id):
    """Mapear ID de cluster a nombre"""
    cluster_names = {
        0: "C2 - Compromiso medio, carga laboral alta",
        1: "C1 - Compromiso alto",
        2: "C3 - Riesgo acumulado",
        None: "Sin asignar"
    }
    return cluster_names.get(cluster_id, "Sin asignar")


def map_risk_to_level(risk_label):
    """Mapear etiqueta de riesgo a nivel numérico"""
    risk_map = {
        'Sin_riesgo': 0,
        'Riesgo_leve': 1,
        'Riesgo_moderado': 2,
        'Riesgo_alto': 3,
        'Riesgo_critico': 4
    }
    return risk_map.get(risk_label, 0)


def format_carga_laboral(carga):
    """Formatear carga laboral para mostrar"""
    if carga == 'No_trabaja':
        return 'No trabaja'
    elif carga == 'Parcial':
        return 'Parcial - 24 h/sem'
    elif carga == 'Completa':
        return 'Completa - 40+ h/sem'
    return 'No especificado'


def format_beca(beca):
    """Formatear tipo de beca"""
    if beca == 'No_tiene':
        return 'Sin beca'
    elif beca == 'Parcial':
        return 'Beca parcial (50%)'
    elif beca == 'Completa':
        return 'Beca completa (100%)'
    return 'No especificado'


def format_deudor(deudor):
    """Formatear estado de deuda"""
    if deudor == 'Sin_deuda':
        return 'Sin deuda activa'
    elif deudor == 'Retraso_leve':
        return 'Retraso leve en pagos'
    elif deudor == 'Retraso_moderado':
        return 'Retraso moderado'
    elif deudor == 'Retraso_crítico':
        return 'Retraso crítico'
    return 'No especificado'


def generar_prediccion_bayesiana(datos_categoricos: dict) -> dict:
    """
    Generar predicción basada en un modelo bayesiano simple
    En producción, esto usaría el modelo ML entrenado
    """
    # Scores para cada variable
    risk_score = 0

    # Análisis de variables de alto impacto
    estres = datos_categoricos.get('estres_academico', 'Moderado')
    if estres in ['Alto', 'Severo', 'Crítico']:
        risk_score += 30
    elif estres == 'Moderado':
        risk_score += 15

    carga_laboral = datos_categoricos.get('carga_laboral', 'No_trabaja')
    if carga_laboral == 'Completa':
        risk_score += 25
    elif carga_laboral == 'Parcial':
        risk_score += 10

    asistencia = datos_categoricos.get('asistencia', 'Frecuente')
    if asistencia in ['Nula', 'Irregular']:
        risk_score += 20

    horas_estudio = datos_categoricos.get('horas_estudio', 'De_1_3h')
    if horas_estudio == 'Menor_a_1h':
        risk_score += 15

    bienestar = datos_categoricos.get('bienestar', 'Moderado')
    if bienestar == 'En_riesgo':
        risk_score += 20

    # Factores protectores (restan riesgo)
    apoyo_familiar = datos_categoricos.get('apoyo_familiar', 'Moderado')
    if apoyo_familiar in ['Fuerte', 'Moderado']:
        risk_score -= 10

    beca = datos_categoricos.get('beca', 'No_tiene')
    if beca in ['Parcial', 'Completa']:
        risk_score -= 5

    # Limitar entre 0 y 100
    risk_score = max(0, min(100, risk_score))

    # Determinar etiqueta de riesgo
    if risk_score < 20:
        risk_label = 'Sin_riesgo'
        cluster = 1  # Compromiso alto
    elif risk_score < 40:
        risk_label = 'Riesgo_leve'
        cluster = 1
    elif risk_score < 60:
        risk_label = 'Riesgo_moderado'
        cluster = 0  # Compromiso medio, carga laboral alta
    elif risk_score < 80:
        risk_label = 'Riesgo_alto'
        cluster = 2  # Riesgo acumulado
    else:
        risk_label = 'Riesgo_critico'
        cluster = 2

    return {
        'risk_level': map_risk_to_level(risk_label),
        'risk_label': risk_label,
        'risk_probability': risk_score,
        'desertion_probability': risk_score,
        'cluster': cluster,
        'cluster_name': get_cluster_name(cluster)
    }


def generar_recomendaciones(student_dict: dict) -> List[str]:
    """Generar recomendaciones dinámicas e inteligentes basadas en el perfil del estudiante"""
    recomendaciones = []

    # Asegurar valores por defecto para evitar NoneType
    riesgo = str(student_dict.get('riesgo_predicho') or '')
    estres = str(student_dict.get('estres_academico') or '')
    carga_laboral = str(student_dict.get('carga_laboral') or '')
    cluster = str(student_dict.get('cluster_asignado') or '')
    apoyo_familiar = str(student_dict.get('apoyo_familiar') or '')
    asistencia = str(student_dict.get('asistencia') or '')
    promedio = float(student_dict.get('promedio_ponderado', 0) or 0)
    horas_estudio = str(student_dict.get('horas_estudio') or '')
    beca = str(student_dict.get('beca') or '')
    deudor = str(student_dict.get('deudor') or '')

    # Normalizar para comparaciones seguras
    riesgo_lower = riesgo.lower()
    estres_norm = estres.lower()
    carga_norm = carga_laboral.lower()
    cluster_norm = cluster.lower()
    apoyo_norm = apoyo_familiar.lower()
    asistencia_norm = asistencia.lower()
    horas_estudio_norm = horas_estudio.lower()
    beca_norm = beca.lower()
    deudor_norm = deudor.lower()

    # 🔴 RIESGO CRÍTICO - Prioridad máxima
    if 'critico' in riesgo_lower or 'crítico' in riesgo_lower:
        recomendaciones.append("🚨 URGENTE: Agendar sesión de intervención inmediata dentro de las próximas 48 horas")
        recomendaciones.append("Activar protocolo de seguimiento intensivo con contacto semanal obligatorio")
        recomendaciones.append("Coordinar con dirección de escuela para plan de contingencia académica")

        if deudor_norm in ['retraso_moderado', 'retraso_crítico', 'retraso_critico']:
            recomendaciones.append("Gestionar urgentemente plan de pagos diferidos con administración")

        if apoyo_norm in ['nulo', 'escaso']:
            recomendaciones.append("Conectar con servicios de asistencia social y apoyo estudiantil externo")

    # 🟠 RIESGO ALTO - Intervención prioritaria
    elif 'alto' in riesgo_lower:
        recomendaciones.append("Agendar sesión de orientación académica prioritaria en los próximos 7 días")
        recomendaciones.append("Iniciar seguimiento quincenal estructurado con objetivos medibles")

        if estres_norm in ['alto', 'severo', 'crítico', 'critico']:
            recomendaciones.append("Derivación a bienestar psicológico para estrategias de manejo de estrés")

        if promedio < 12:
            recomendaciones.append("Implementar plan de reforzamiento académico con tutorías especializadas")

    # 🟡 RIESGO MODERADO - Monitoreo activo
    elif 'moderado' in riesgo_lower:
        recomendaciones.append("Programar sesión de seguimiento académico en las próximas 2 semanas")
        recomendaciones.append("Implementar sistema de alertas tempranas para prevenir escalada de riesgo")

    # 🔵 RIESGO LEVE - Seguimiento preventivo
    elif 'leve' in riesgo_lower:
        recomendaciones.append("Mantener seguimiento mensual preventivo y reforzar factores protectores")

    # ✅ SIN RIESGO - Refuerzo positivo
    else:
        recomendaciones.append("Mantener el excelente desempeño actual con reconocimiento positivo")
        recomendaciones.append("Invitar a participar como mentor/tutor par para otros estudiantes")

    # RECOMENDACIONES SEGÚN CLÚSTER
    if 'c1' in cluster_norm:  # Compromiso alto
        if promedio >= 14:
            recomendaciones.append("Motivar participación en proyectos de investigación o actividades extracurriculares")

    elif 'c2' in cluster_norm:  # Estrés académico / Carga laboral alta
        recomendaciones.append("Evaluar redistribución de carga académica o considerar reducción de créditos")

        if carga_norm == 'completa':
            recomendaciones.append("Negociar con empleador flexibilidad horaria durante semanas de exámenes")
            recomendaciones.append("Explorar opciones de práctica preprofesional para validar horas laborales")

    elif 'c3' in cluster_norm:  # Riesgo acumulado / Crítico
        recomendaciones.append("Activar red de soporte integral: académico, emocional y económico")
        recomendaciones.append("Considerar retiro temporal estratégico si la salud mental está comprometida")

    # FACTORES ESPECÍFICOS ADICIONALES

    # Asistencia irregular o baja
    if asistencia_norm in ['nula', 'irregular']:
        recomendaciones.append("Identificar barreras de asistencia (transporte, salud, económicas) y buscar soluciones")
        recomendaciones.append("Implementar sistema de recordatorios y acompañamiento para mejorar asistencia")

    # Pocas horas de estudio
    if horas_estudio_norm == 'menor_a_1h' or 'menor' in horas_estudio_norm:
        recomendaciones.append("Desarrollar plan de gestión del tiempo con bloques de estudio de 25-50 minutos")
        recomendaciones.append("Enseñar técnicas de estudio efectivas: Pomodoro, resúmenes activos, mapas conceptuales")

    # Carga laboral completa
    if carga_norm == 'completa':
        recomendaciones.append("Explorar becas de estudio o programas de apoyo económico para reducir horas laborales")

    # Sin apoyo familiar
    if apoyo_norm in ['nulo', 'escaso']:
        recomendaciones.append("Conectar con grupos de apoyo estudiantil y construir red de soporte alternativa")
        recomendaciones.append("Informar sobre servicios de residencia estudiantil o apoyo habitacional si aplica")

    # Problemas económicos (sin beca + deudor)
    if beca_norm == 'no_tiene' and deudor_norm in ['retraso_moderado', 'retraso_crítico', 'retraso_critico']:
        recomendaciones.append("Gestionar evaluación socioeconómica urgente para acceso a becas de emergencia")
        recomendaciones.append("Informar sobre programas de trabajo universitario o asistencia alimentaria")

    # Estrés elevado
    if estres_norm in ['alto', 'severo', 'crítico', 'critico']:
        recomendaciones.append("Promover técnicas de autocuidado: mindfulness, ejercicio regular, pausas activas")
        recomendaciones.append("Evaluar sobrecarga académica y priorizar cursos esenciales vs. electivos")

    # Si no hay recomendaciones específicas, agregar seguimiento estándar
    if not recomendaciones:
        recomendaciones.append("Continuar con seguimiento regular según protocolo institucional")
        recomendaciones.append("Reforzar comunicación abierta y disponibilidad para consultas")

    return recomendaciones


def generar_factores_clave(student_dict: dict) -> List[Dict[str, str]]:
    """Generar factores clave del riesgo"""
    factores = []

    # Asegurar valores por defecto para evitar NoneType
    estres = str(student_dict.get('estres_academico') or '')
    carga_laboral = str(student_dict.get('carga_laboral') or '')
    apoyo_familiar = str(student_dict.get('apoyo_familiar') or '')

    estres_norm = estres.lower()
    carga_norm = carga_laboral.lower()
    apoyo_norm = apoyo_familiar.lower()

    if estres_norm in ['alto', 'severo', 'crítico', 'critico']:
        factores.append({
            'factor': 'Estrés académico',
            'nivel': 'Alto impacto',
            'descripcion': f'Nivel de estrés: {estres or "No especificado"}'
        })

    if carga_norm in ['parcial', 'completa']:
        factores.append({
            'factor': 'Carga laboral',
            'nivel': 'Incrementa el riesgo',
            'descripcion': f'Carga laboral: {carga_laboral or "No especificado"}'
        })

    if apoyo_norm in ['fuerte', 'moderado']:
        factores.append({
            'factor': 'Apoyo familiar',
            'nivel': 'Factor protector',
            'descripcion': f'Apoyo: {apoyo_familiar or "No especificado"}'
        })

    return factores


if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando servidor...")
    print("📊 Conectando a PostgreSQL...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
