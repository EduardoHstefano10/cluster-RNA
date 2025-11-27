#!/bin/bash

# Script de inicialización de la base de datos
# Este script crea la base de datos, usuario y tablas necesarias

echo "🔧 Configurando Base de Datos para Sistema de Alerta Temprana"
echo "============================================================="

# Cargar variables de entorno
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

DB_NAME=${DB_NAME:-estudiantes_db}
DB_USER=${DB_USER:-cluster_user}
DB_PASSWORD=${DB_PASSWORD:-cluster_pass_2024}
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}

echo "📋 Configuración:"
echo "  Base de datos: $DB_NAME"
echo "  Usuario: $DB_USER"
echo "  Host: $DB_HOST"
echo "  Puerto: $DB_PORT"
echo ""

# Verificar si PostgreSQL está corriendo
echo "🔍 Verificando PostgreSQL..."
if ! pg_isready -h $DB_HOST -p $DB_PORT > /dev/null 2>&1; then
    echo "❌ PostgreSQL no está corriendo en $DB_HOST:$DB_PORT"
    echo "   Inicia PostgreSQL con: sudo service postgresql start"
    exit 1
fi
echo "✅ PostgreSQL está corriendo"

# Crear usuario y base de datos si no existen
echo ""
echo "👤 Creando usuario y base de datos..."

# Intentar conectarse como postgres
sudo -u postgres psql -c "SELECT 1" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    # Crear usuario
    sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Usuario $DB_USER creado"
    else
        echo "ℹ️  Usuario $DB_USER ya existe"
    fi

    # Crear base de datos
    sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Base de datos $DB_NAME creada"
    else
        echo "ℹ️  Base de datos $DB_NAME ya existe"
    fi

    # Dar permisos
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
    echo "✅ Permisos otorgados"
else
    echo "❌ No se pudo conectar como usuario postgres"
    echo "   Intenta ejecutar manualmente:"
    echo "   sudo -u postgres createuser -P $DB_USER"
    echo "   sudo -u postgres createdb -O $DB_USER $DB_NAME"
fi

# Inicializar tablas
echo ""
echo "📊 Inicializando tablas..."
if [ -f init.sql ]; then
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f init.sql
    if [ $? -eq 0 ]; then
        echo "✅ Tablas creadas exitosamente"
    else
        echo "❌ Error al crear tablas"
        exit 1
    fi
else
    echo "❌ No se encontró el archivo init.sql"
    exit 1
fi

# Verificar tablas creadas
echo ""
echo "🔍 Verificando tablas..."
TABLES=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT tablename FROM pg_tables WHERE schemaname='public';")
echo "Tablas creadas:"
echo "$TABLES"

echo ""
echo "✅ ¡Base de datos configurada exitosamente!"
echo ""
echo "🚀 Puedes iniciar el servidor con: python main.py"
