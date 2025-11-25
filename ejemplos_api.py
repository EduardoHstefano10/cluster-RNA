"""
Ejemplos de uso de la API del Sistema de Alerta Temprana
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def ejemplo_prediccion():
    """Ejemplo: Predecir riesgo de un estudiante"""
    print("=" * 60)
    print("EJEMPLO 1: Predicción de Riesgo Académico")
    print("=" * 60)

    # Datos del estudiante
    estudiante = {
        "Promedio_ponderado": 15.5,
        "Creditos_matriculados": 20,
        "Porcentaje_creditos_aprobados": 75,
        "Cursos_desaprobados": 1,
        "Asistencia": 87,
        "Retiros_cursos": 1,
        "Edad": 21,
        "Horas_trabajo_semana": 15,
        "Anio_ingreso": 2015,
        "Numero_ciclos_academicos": 10,
        "Cursos_matriculados_ciclo": 6,
        "Horas_estudio_semana": 17,
        "indice_regularidad": 65,
        "Intentos_aprobacion_curso": 1,
        "Nota_promedio": 16
    }

    # Hacer la predicción
    response = requests.post(f"{BASE_URL}/api/predict", json=estudiante)
    resultado = response.json()

    print(f"\n📊 Resultado de la Predicción:")
    print(f"   Nivel de Riesgo: {resultado['risk_label']}")
    print(f"   Probabilidad: {resultado['risk_probability']:.2f}%")
    print(f"   Probabilidad de Deserción: {resultado['desertion_probability']:.2f}%")
    print(f"   Clúster Asignado: {resultado['cluster_name']}")

    print(f"\n💡 Recomendaciones:")
    for i, rec in enumerate(resultado['recommendations'], 1):
        print(f"   {i}. {rec}")

    print(f"\n🔍 Factores Clave:")
    for factor in resultado['key_factors']:
        print(f"   • {factor['factor']}: {factor['nivel']}")
        print(f"     {factor['descripcion']}")


def ejemplo_listar_estudiantes():
    """Ejemplo: Obtener lista de estudiantes"""
    print("\n" + "=" * 60)
    print("EJEMPLO 2: Listar Estudiantes")
    print("=" * 60)

    # Obtener estudiantes
    response = requests.get(f"{BASE_URL}/api/students?limit=5")
    data = response.json()

    print(f"\n📋 Total de estudiantes: {data['total']}")
    print(f"   Mostrando: {data['showing']}\n")

    for estudiante in data['students']:
        print(f"   👤 {estudiante['nombre']}")
        print(f"      Código: {estudiante['codigo']}")
        print(f"      Carrera: {estudiante['carrera']}")
        print(f"      Promedio: {estudiante['promedio']}/20")
        print(f"      Riesgo: {estudiante['riesgo_predicho']}")
        print(f"      Clúster: {estudiante['cluster_asignado']}")
        print()


def ejemplo_estadisticas():
    """Ejemplo: Obtener estadísticas del dashboard"""
    print("=" * 60)
    print("EJEMPLO 3: Estadísticas del Sistema")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/api/stats")
    stats = response.json()

    print(f"\n📊 Estadísticas Generales:")
    print(f"   Total de estudiantes: {stats['total_estudiantes']}")
    print(f"   Precisión del modelo: {stats['precision_modelo']}%")
    print(f"   Estudiantes en alto riesgo: {stats['estudiantes_alto_riesgo']}")
    print(f"   En seguimiento activo: {stats['seguimiento_activo']}")
    print(f"   Número de clústeres: {stats['num_clusters']}")

    print(f"\n🏷️  Clústeres Activos:")
    for cluster in stats['clusters_activos']:
        print(f"   • {cluster}")


def ejemplo_perfil_estudiante():
    """Ejemplo: Obtener perfil completo de un estudiante"""
    print("\n" + "=" * 60)
    print("EJEMPLO 4: Perfil del Estudiante")
    print("=" * 60)

    codigo = "20231547"
    response = requests.get(f"{BASE_URL}/api/students/{codigo}")
    data = response.json()

    estudiante = data['student']
    prediccion = data['prediction']
    resumen = data['resumen_academico']

    print(f"\n👤 {estudiante['nombre']}")
    print(f"   Código: {estudiante['codigo']}")
    print(f"   Carrera: {estudiante['carrera']}")
    print(f"   Ciclo: {estudiante['ciclo']}")

    print(f"\n📚 Resumen Académico:")
    print(f"   Promedio Ponderado: {resumen['promedio_ponderado']}/20")
    print(f"   Créditos Cursados: {resumen['creditos_cursados']}")
    print(f"   Asistencia (4 sem): {resumen['asistencia_ultimas_4_semanas']}")

    print(f"\n⚠️  Predicción de Riesgo:")
    print(f"   Nivel: {prediccion['risk_label']}")
    print(f"   Probabilidad: {prediccion['risk_probability']:.2f}%")
    print(f"   Clúster: {prediccion['cluster_name']}")


def ejemplo_filtrar_estudiantes():
    """Ejemplo: Filtrar estudiantes por riesgo y clúster"""
    print("\n" + "=" * 60)
    print("EJEMPLO 5: Filtrar Estudiantes")
    print("=" * 60)

    # Filtrar por riesgo alto
    print("\n🔴 Estudiantes en Alto Riesgo:")
    response = requests.get(f"{BASE_URL}/api/students?riesgo=alto&limit=3")
    data = response.json()

    for estudiante in data['students']:
        print(f"   • {estudiante['nombre']} - {estudiante['riesgo_predicho']}")

    # Filtrar por clúster
    print("\n📊 Estudiantes del Clúster 2:")
    response = requests.get(f"{BASE_URL}/api/students?cluster=2&limit=3")
    data = response.json()

    for estudiante in data['students']:
        print(f"   • {estudiante['nombre']} - {estudiante['cluster_asignado']}")


def ejemplo_registrar_estudiante():
    """Ejemplo: Registrar un nuevo estudiante"""
    print("\n" + "=" * 60)
    print("EJEMPLO 6: Registrar Nuevo Estudiante")
    print("=" * 60)

    nuevo_estudiante = {
        "codigo": "20241234",
        "nombre": "Juan Pérez García",
        "carrera": "Ingeniería de Sistemas",
        "ciclo": 3,
        "datos": {
            "Promedio_ponderado": 16.5,
            "Creditos_matriculados": 22,
            "Porcentaje_creditos_aprobados": 82,
            "Cursos_desaprobados": 0,
            "Asistencia": 92,
            "Retiros_cursos": 0,
            "Edad": 20,
            "Horas_trabajo_semana": 10,
            "Anio_ingreso": 2022,
            "Numero_ciclos_academicos": 5,
            "Cursos_matriculados_ciclo": 7,
            "Horas_estudio_semana": 20,
            "indice_regularidad": 75,
            "Intentos_aprobacion_curso": 1,
            "Nota_promedio": 17
        }
    }

    response = requests.post(
        f"{BASE_URL}/api/students/register",
        json=nuevo_estudiante
    )
    resultado = response.json()

    print(f"\n✅ {resultado['message']}")
    print(f"   ID Estudiante: {resultado['student_id']}")
    print(f"   Riesgo: {resultado['prediction']['risk_label']}")
    print(f"   Clúster: {resultado['prediction']['cluster_name']}")


def ejemplo_cluster_info():
    """Ejemplo: Obtener información de un clúster"""
    print("\n" + "=" * 60)
    print("EJEMPLO 7: Información de Clústeres")
    print("=" * 60)

    for cluster_id in [0, 1, 2]:
        response = requests.get(f"{BASE_URL}/api/clusters/{cluster_id}")
        cluster = response.json()

        print(f"\n🏷️  {cluster['name']}")
        print(f"   Descripción: {cluster['description']}")
        print(f"   Riesgo Promedio: Nivel {cluster['avg_risk']}")
        print(f"   Tamaño: {cluster['size']} estudiantes")
        print(f"   Características:")
        for key, value in cluster['characteristics'].items():
            print(f"      • {key}: {value}")


# Función principal para ejecutar todos los ejemplos
def main():
    print("\n" + "=" * 60)
    print("🎓 SISTEMA DE ALERTA TEMPRANA - EJEMPLOS DE USO")
    print("=" * 60)

    try:
        # Verificar que el servidor esté corriendo
        response = requests.get(f"{BASE_URL}/api/stats", timeout=2)
        if response.status_code != 200:
            raise Exception("Servidor no disponible")

        print("\n✅ Servidor conectado correctamente\n")

        # Ejecutar ejemplos
        ejemplo_prediccion()
        ejemplo_listar_estudiantes()
        ejemplo_estadisticas()
        ejemplo_perfil_estudiante()
        ejemplo_filtrar_estudiantes()
        ejemplo_registrar_estudiante()
        ejemplo_cluster_info()

        print("\n" + "=" * 60)
        print("✅ Todos los ejemplos ejecutados exitosamente")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: No se pudo conectar al servidor")
        print("   Asegúrate de que el servidor esté corriendo:")
        print("   python main.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
