# 📚 Sistema de Alerta Temprana - Instrucciones de Uso

## 🚀 Inicio Rápido

### 1. Iniciar el Servidor

El servidor ya está corriendo en: **http://localhost:8000**

Si necesitas iniciarlo manualmente:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Acceder al Sistema

- **Formulario de Registro**: http://localhost:8000
- **Panel del Tutor**: http://localhost:8000/panel
- **Documentación API**: http://localhost:8000/docs

---

## 📝 Funcionalidades Implementadas

### ✅ 1. Registro de Estudiantes con Predicción Automática

El formulario de registro ahora:
- **Guarda automáticamente** los datos del estudiante
- **Genera predicción automática** usando la Red Neuronal Artificial (RNA)
- **Asigna cluster** automáticamente usando K-means
- **Calcula probabilidad de deserción**
- **Genera recomendaciones personalizadas**

**Cómo usar:**
1. Ir a http://localhost:8000
2. Llenar el formulario con los datos del estudiante:
   - Código del estudiante
   - Nombre completo
   - Edad
   - Carrera
   - Ciclo
3. Completar las 15 variables numéricas:
   - Promedio ponderado (0-20)
   - Créditos matriculados
   - Porcentaje de créditos aprobados
   - Cursos desaprobados
   - Asistencia (%)
   - Y más...
4. Hacer clic en **"Cargar datos del estudiante"**
5. El sistema automáticamente:
   - Guarda el estudiante
   - Genera la predicción
   - Muestra el riesgo predicho
   - Muestra el cluster asignado
   - Muestra recomendaciones

### ✅ 2. Base de Datos Inteligente

El sistema ahora soporta dos modos:

**Modo PostgreSQL** (cuando la base de datos está disponible):
- Guarda todos los estudiantes en PostgreSQL
- Guarda las predicciones en la base de datos
- Permite consultas y análisis persistentes

**Modo Memoria** (fallback automático):
- Si PostgreSQL no está disponible, usa almacenamiento en memoria
- Todos los datos se guardan temporalmente
- Ideal para desarrollo y pruebas

### ✅ 3. Predicción con RNA y Clustering

El modelo de predicción incluye:

**Red Neuronal Artificial (MLP):**
- Arquitectura: 64-32-16 neuronas
- 5 niveles de riesgo:
  - 0: Sin riesgo
  - 1: Riesgo leve
  - 2: Riesgo moderado
  - 3: Riesgo alto
  - 4: Riesgo crítico
- Precisión: ~92.4%

**Clustering K-means:**
- 3 clusters identificados:
  - **C1**: Compromiso alto (850 estudiantes)
  - **C2**: Estrés académico (650 estudiantes)
  - **C3**: Riesgo acumulado (500 estudiantes)

---

## 🗄️ Configuración de PostgreSQL (Opcional)

### Iniciar PostgreSQL

Para usar PostgreSQL, necesitas iniciarlo:

**Opción 1: Docker (Recomendado)**
```bash
docker-compose up -d
```

**Opción 2: Servicio local**
```bash
sudo service postgresql start
```

### Cargar Datos del CSV a PostgreSQL

Una vez PostgreSQL esté corriendo, carga los datos:

```bash
python load_csv_to_db.py
```

Este script:
- Lee el CSV de `data/estudiantes_data.csv`
- Convierte las 15 variables numéricas a categorías
- Inserta 2000 estudiantes en PostgreSQL
- Genera valores categóricos inteligentes basados en los datos numéricos

---

## 📊 Estructura de Datos

### Entrada del Formulario (15 variables numéricas)

```json
{
  "Promedio_ponderado": 15.5,
  "Creditos_matriculados": 22,
  "Porcentaje_creditos_aprobados": 85.0,
  "Cursos_desaprobados": 1,
  "Asistencia": 90.0,
  "Retiros_cursos": 0,
  "Edad": 20,
  "Horas_trabajo_semana": 15.0,
  "Anio_ingreso": 2021,
  "Numero_ciclos_academicos": 6,
  "Cursos_matriculados_ciclo": 7,
  "Horas_estudio_semana": 18.0,
  "indice_regularidad": 75.0,
  "Intentos_aprobacion_curso": 1,
  "Nota_promedio": 15.2
}
```

### Salida de la Predicción

```json
{
  "risk_level": 1,
  "risk_label": "Riesgo_leve",
  "risk_probability": 0.85,
  "desertion_probability": 15.3,
  "cluster": 1,
  "cluster_name": "C1 - Compromiso alto",
  "recommendations": [
    "Mantener el buen desempeño actual",
    "Continuar con seguimiento regular"
  ],
  "key_factors": [
    {
      "factor": "Asistencia",
      "nivel": "Factor protector",
      "descripcion": "Buena regularidad en asistencia"
    }
  ]
}
```

---

## 🔍 Panel del Tutor

Accede a http://localhost:8000/panel para:
- Ver listado de todos los estudiantes
- Filtrar por nivel de riesgo
- Filtrar por cluster
- Filtrar por estado de seguimiento
- Exportar datos a CSV
- Ver estadísticas generales

---

## 🎯 Flujo de Trabajo Recomendado

1. **Registrar estudiante nuevo**
   - Llenar formulario
   - Sistema guarda y predice automáticamente

2. **Ver predicción**
   - Revisar nivel de riesgo
   - Revisar cluster asignado
   - Leer recomendaciones

3. **Tomar acción**
   - Si riesgo alto/crítico: Agendar tutoría
   - Si riesgo moderado: Monitorear
   - Si sin riesgo: Seguimiento regular

4. **Actualizar datos**
   - Buscar estudiante por código
   - Actualizar información
   - Nueva predicción automática

---

## 📈 Interpretación de Clusters

### C1 - Compromiso Alto
- **Características**: Alta asistencia, bajo estrés, muchas horas de estudio
- **Riesgo promedio**: Bajo (0)
- **Acción**: Seguimiento regular

### C2 - Estrés Académico
- **Características**: Buena asistencia pero alta carga laboral
- **Riesgo promedio**: Moderado (2)
- **Acción**: Apoyo en gestión de tiempo

### C3 - Riesgo Acumulado
- **Características**: Baja asistencia, múltiples factores de riesgo
- **Riesgo promedio**: Alto (3)
- **Acción**: Intervención inmediata

---

## 🛠️ Solución de Problemas

### El servidor no inicia
```bash
# Instalar dependencias
pip install -r requirements.txt

# Iniciar servidor
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### PostgreSQL no conecta
- El sistema automáticamente usa almacenamiento en memoria
- Para usar PostgreSQL, asegúrate de que esté corriendo:
  ```bash
  # Verificar
  pg_isready -h localhost -p 5432

  # Iniciar
  docker-compose up -d
  ```

### No se carga la predicción
- Verifica que el modelo esté en `models/neural_network.pkl`
- Si no existe, el sistema lo entrenará automáticamente

---

## 🎓 Ejemplos de Uso

### Ejemplo 1: Estudiante de Bajo Riesgo

**Entrada:**
- Promedio: 17.5
- Asistencia: 95%
- Cursos desaprobados: 0
- Horas estudio: 20h/semana

**Predicción Esperada:**
- Riesgo: Sin riesgo o Leve
- Cluster: C1 (Compromiso alto)
- Recomendación: Mantener buen desempeño

### Ejemplo 2: Estudiante de Alto Riesgo

**Entrada:**
- Promedio: 11.5
- Asistencia: 70%
- Cursos desaprobados: 3
- Horas trabajo: 40h/semana

**Predicción Esperada:**
- Riesgo: Alto o Crítico
- Cluster: C3 (Riesgo acumulado)
- Recomendaciones:
  - Agendar sesión de orientación prioritaria
  - Coordinar apoyo psicológico
  - Considerar reducción de carga académica

---

## 📞 API Endpoints

### Registrar Estudiante
```bash
POST /api/students/register
Content-Type: application/json

{
  "codigo": "20231547",
  "nombre": "Ana Castillo",
  "carrera": "Ingeniería",
  "ciclo": 5,
  "datos": { ... }
}
```

### Obtener Predicción
```bash
GET /api/students/20231547
```

### Estadísticas
```bash
GET /api/stats
```

---

## ✨ Características Destacadas

1. ✅ **Predicción automática** al registrar estudiante
2. ✅ **Guardado en base de datos** con predicción incluida
3. ✅ **Clustering automático** para identificar perfiles
4. ✅ **Recomendaciones personalizadas** basadas en cluster y riesgo
5. ✅ **Fallback a memoria** si PostgreSQL no está disponible
6. ✅ **Modelos pre-entrenados** listos para usar
7. ✅ **Panel de tutor** con filtros y exportación
8. ✅ **2000 estudiantes de ejemplo** en el CSV

---

## 🎯 Próximos Pasos

Para mejorar el sistema, puedes:

1. **Iniciar PostgreSQL** para persistencia de datos
2. **Cargar CSV completo** con `python load_csv_to_db.py`
3. **Agregar visualizaciones** de clusters en el frontend
4. **Personalizar recomendaciones** según tu institución
5. **Ajustar umbrales** de riesgo según necesidades

---

## 📝 Notas Importantes

- El sistema está **completamente funcional** sin PostgreSQL
- Las predicciones son **automáticas** al guardar estudiante
- Los datos en memoria se **pierden al reiniciar** el servidor
- Para persistencia, usa PostgreSQL
- El modelo tiene **92.4% de precisión** en el conjunto de prueba

---

¡El sistema está listo para usar! 🎉
