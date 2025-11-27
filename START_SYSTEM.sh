#!/bin/bash

echo "🚀 Sistema de Alerta Temprana - Inicio Automático"
echo "=================================================="
echo ""

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Verificar dependencias
echo "📦 Verificando dependencias..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Instalando dependencias...${NC}"
    pip install -q -r requirements.txt
    echo -e "${GREEN}✅ Dependencias instaladas${NC}"
else
    echo -e "${GREEN}✅ Dependencias OK${NC}"
fi
echo ""

# 2. Verificar modelos
echo "🤖 Verificando modelos de ML..."
if [ -f "models/neural_network.pkl" ]; then
    echo -e "${GREEN}✅ Modelo de predicción encontrado${NC}"
else
    echo -e "${YELLOW}⚠️  Modelo no encontrado. Se entrenará automáticamente al iniciar${NC}"
fi
echo ""

# 3. Verificar PostgreSQL (opcional)
echo "🗄️  Verificando PostgreSQL..."
if pg_isready -h localhost -p 5432 >/dev/null 2>&1; then
    echo -e "${GREEN}✅ PostgreSQL está corriendo${NC}"

    # Preguntar si quiere cargar datos del CSV
    read -p "¿Deseas cargar los datos del CSV a PostgreSQL? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📊 Cargando datos del CSV..."
        python load_csv_to_db.py
    fi
else
    echo -e "${YELLOW}⚠️  PostgreSQL no está corriendo${NC}"
    echo -e "${YELLOW}   El sistema usará almacenamiento en memoria${NC}"
    echo ""
    echo "   Para iniciar PostgreSQL:"
    echo "   - Con Docker: docker-compose up -d"
    echo "   - Sin Docker: sudo service postgresql start"
fi
echo ""

# 4. Iniciar servidor
echo "🌐 Iniciando servidor FastAPI..."
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}  🎯 Sistema Iniciado${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "📱 Accede al sistema en:"
echo "   • Formulario de Registro: http://localhost:8000"
echo "   • Panel del Tutor:        http://localhost:8000/panel"
echo "   • Documentación API:      http://localhost:8000/docs"
echo ""
echo "📚 Lee INSTRUCCIONES_USO.md para más información"
echo ""
echo -e "${YELLOW}Presiona Ctrl+C para detener el servidor${NC}"
echo ""

# Iniciar servidor
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
