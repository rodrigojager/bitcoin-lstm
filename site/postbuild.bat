@echo off
echo 🔧 Empacotando React (TSX) com esbuild...

set NODE_PATH=%~dp0node_modules
build\esbuild.exe wwwroot\react\index.tsx ^
  --bundle ^
  --outfile=wwwroot\assets\js\bundle.js ^
  --format=iife ^
  --jsx=automatic

if %errorlevel% neq 0 (
    echo ❌ Erro ao empacotar React
    exit /b %errorlevel%
)

echo ✅ Bundle gerado com sucesso!
