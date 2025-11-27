@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 Sistema de Alerta Temprana - Inicio Automático
echo ==================================================
echo.

REM 1. Verificar dependencias
echo 📦 Verificando dependencias...
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo ⚠️  Instalando dependencias...
    pip install -q -r requirements.txt
    echo ✅ Dependencias instaladas
) else (
    echo ✅ Dependencias OK
)
echo.

REM 2. Verificar modelos
echo 🤖 Verificando modelos de ML...
if exist "models\neural_network.pkl" (
    echo ✅ Modelo de predicción encontrado
) else (
    echo ⚠️  Modelo no encontrado. Se entrenará automáticamente al iniciar
)
echo.

REM 3. Verificar PostgreSQL (opcional)
echo 🗄️  Verificando PostgreSQL...
pg_isready -h localhost -p 5432 >nul 2>&1
if errorlevel 1 (
    echo ⚠️  PostgreSQL no está corriendo
    echo    El sistema usará almacenamiento en memoria
    echo.
    echo    Para iniciar PostgreSQL:
    echo    - Con Docker: docker-compose up -d
    echo    - Sin Docker: Inicia el servicio PostgreSQL desde servicios de Windows
) else (
    echo ✅ PostgreSQL está corriendo
    echo.
    set /p REPLY="¿Deseas cargar los datos del CSV a PostgreSQL? (y/n) "
    if /i "!REPLY!"=="y" (
        echo 📊 Cargando datos del CSV...
        python load_csv_to_db.py
    )
)
echo.

REM 4. Iniciar servidor
echo 🌐 Iniciando servidor FastAPI...
echo.
echo ================================
echo   🎯 Sistema Iniciado
echo ================================
echo.
echo 📱 Accede al sistema en:
echo    • Formulario de Registro: http://localhost:8000
echo    • Panel del Tutor:        http://localhost:8000/panel
echo    • Documentación API:      http://localhost:8000/docs
echo.
echo 📚 Lee INSTRUCCIONES_USO.md para más información
echo.
echo Presiona Ctrl+C para detener el servidor
echo.

REM Iniciar servidor
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
