# 🚀 Guía Rápida de Inicio

## Ejecutar el Sistema en 3 Pasos

### 1️⃣ Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2️⃣ (Opcional) Entrenar el modelo
Si no existen los modelos o quieres reentrenar:
```bash
python ml_models.py
```

**Nota**: Los modelos ya están entrenados en la carpeta `models/`

### 3️⃣ Iniciar el servidor
```bash
python main.py
```

O con uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🌐 Acceder a la Aplicación

Abre tu navegador en: **http://localhost:8000**

### Rutas disponibles:
- **http://localhost:8000** → Registro/Actualizar Estudiante
- **http://localhost:8000/panel** → Panel del Tutor
- **http://localhost:8000/perfil/20231547** → Perfil del Estudiante
- **http://localhost:8000/docs** → Documentación interactiva de la API (Swagger)

## 📊 Probar la API

### Predecir riesgo de un estudiante:
```bash
curl -X POST "http://localhost:8000/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

### Obtener lista de estudiantes:
```bash
curl "http://localhost:8000/api/students?limit=5"
```

### Obtener estadísticas:
```bash
curl "http://localhost:8000/api/stats"
```

## 🎯 Características Principales

✅ **Predicción de Riesgo Académico** con RNA
- 5 niveles: Sin riesgo, Leve, Moderado, Alto, Crítico
- Precisión: 83.75%

✅ **Clustering de Estudiantes**
- 3 clústeres: C1 (Compromiso alto), C2 (Estrés académico), C3 (Riesgo acumulado)

✅ **Interfaz Web Moderna**
- Bootstrap 5
- Responsive design
- Gráficos interactivos con Chart.js

✅ **API RESTful Completa**
- 15+ endpoints
- Documentación automática (Swagger)
- CORS habilitado

## 🔧 Solución de Problemas

### Error: "No module named 'fastapi'"
```bash
pip install fastapi uvicorn
```

### Error: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Puerto 8000 ocupado
Cambia el puerto:
```bash
uvicorn main:app --port 3000
```

### Los modelos no existen
Ejecuta:
```bash
python ml_models.py
```

## 📚 Recursos

- 📖 [README.md](README.md) - Documentación completa
- 🌐 [FastAPI Docs](http://localhost:8000/docs) - API interactiva
- 📊 [Notebook](PCAGRUPAL.ipynb) - Análisis exploratorio

## 💡 Tips

1. **Modo desarrollo**: Usa `--reload` para recargar automáticamente
2. **Ver logs**: Agrega `--log-level debug`
3. **Producción**: Usa `--workers 4` para múltiples procesos

---

**¡Disfruta explorando el sistema! 🎓**
