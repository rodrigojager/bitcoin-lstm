@echo off
echo 🛠️  Verificando dependências NPM...

where npm >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js / npm não está instalado. Instale em: https://nodejs.org
    exit /b 1
)

if exist "%~dp0node_modules" (
    echo ✅ Dependências já estão instaladas.
) else (
    echo 📦 Instalando dependências NPM...
    call npm install
    call npx tsc --noEmit
    if %errorlevel% neq 0 (
        echo ❌ Falha ao instalar dependências NPM.
        exit /b %errorlevel%
    )
)

@echo off
rem Chama o script de credenciais DEV, se existir
if exist "%SolutionDir%dev_env.bat" (
    call "%SolutionDir%dev_env.bat"
) else (
    echo [INFO] dev_env.bat nao encontrado - prosseguindo sem alterar variaveis.
)