#!/bin/bash

# load_emscripten.sh
#source /home/marcos/emsdk/emsdk_env.sh

#source /home/marcos/Documents/chess_v7/chess_v6/nn/train_export_bias/new_train/nnue_env/bin/activate

#/home/marcos/Documents/codes/mysites/chess_130826_b

# Script para carregar o ambiente Emscripten no shell atual

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para imprimir mensagens
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Função para carregar o ambiente Emscripten de um diretório específico
load_emscripten_from_path() {
    local emsdk_path="${1:-/home/marcos/emsdk}"
    
    print_step "Carregando ambiente Emscripten de: $emsdk_path"
    
    # Verificar se o diretório existe
    if [ ! -d "$emsdk_path" ]; then
        print_error "Diretório $emsdk_path não encontrado!"
        print_message "Verifique se o caminho está correto."
        return 1
    fi
    
    # Verificar se o script emsdk_env.sh existe
    if [ ! -f "$emsdk_path/emsdk_env.sh" ]; then
        print_error "Arquivo emsdk_env.sh não encontrado em $emsdk_path"
        print_message "Certifique-se de que o Emscripten está instalado corretamente."
        return 1
    fi
    
    # Salvar o diretório atual
    local current_dir=$(pwd)
    
    # Mudar para o diretório do Emsdk
    cd "$emsdk_path"
    
    print_message "Carregando ambiente Emscripten..."
    
    # Carregar o ambiente
    source ./emsdk_env.sh
    
    # Voltar para o diretório original
    cd "$current_dir"
    
    # Verificar se as variáveis foram carregadas
    if [ -n "$EMSDK" ]; then
        print_message "✅ Ambiente Emscripten carregado com sucesso!"
        print_message "  EMSDK: $EMSDK"
        print_message "  EM_CONFIG: $EM_CONFIG"
        print_message "  PATH inclui: $(which emcc 2>/dev/null || echo 'emcc não encontrado no PATH')"
        return 0
    else
        print_warning "⚠️  Parece que o ambiente não foi carregado corretamente."
        print_message "Tente executar manualmente: source $emsdk_path/emsdk_env.sh"
        return 1
    fi
}

# Função para verificar a instalação
verify_emscripten() {
    print_step "Verificando instalação do Emscripten"
    
    # Verificar comandos principais
    local commands=("emcc" "em++" "emar" "embuilder")
    local all_ok=true
    
    for cmd in "${commands[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            if [ "$cmd" = "emcc" ] || [ "$cmd" = "em++" ]; then
                print_message "  $cmd: $(eval $cmd --version | head -n1)"
            else
                print_message "  $cmd: ✓ Disponível"
            fi
        else
            print_warning "  $cmd: ❌ Não encontrado"
            all_ok=false
        fi
    done
    
    if [ "$all_ok" = true ]; then
        print_message "✅ Todos os comandos Emscripten estão disponíveis!"
    else
        print_warning "⚠️  Alguns comandos não foram encontrados no PATH."
        print_message "Certifique-se de que o ambiente foi carregado corretamente."
    fi
}

# Função para testar uma compilação simples
test_compilation() {
    print_step "Testando compilação simples"
    
    # Criar um arquivo C simples para teste
    local test_file="/tmp/test_emscripten.c"
    local output_file="/tmp/test_emscripten.html"
    
    cat > "$test_file" << 'EOF'
#include <stdio.h>
#include <emscripten/emscripten.h>

int main() {
    printf("Hello from Emscripten!\n");
    printf("Emscripten version: %d.%d.%d\n", 
           __EMSCRIPTEN_major__, 
           __EMSCRIPTEN_minor__, 
           __EMSCRIPTEN_tiny__);
    return 0;
}
EOF
    
    print_message "Compilando arquivo de teste: $test_file"
    
    if emcc "$test_file" -o "$output_file" -s WASM=1 -s EXPORTED_FUNCTIONS='["_main"]' 2>/dev/null; then
        print_message "✅ Compilação bem-sucedida!"
        print_message "  Arquivo gerado: $output_file"
        print_message "  Tamanho: $(ls -lh "$output_file" | awk '{print $5}')"
        rm -f "$test_file" "$output_file"
        return 0
    else
        print_error "❌ Falha na compilação de teste."
        rm -f "$test_file"
        return 1
    fi
}

# Função para mostrar informações do ambiente
show_environment() {
    print_step "Informações do Ambiente Emscripten"
    
    echo "EMSDK: ${EMSDK:-'Não definido'}"
    echo "EM_CONFIG: ${EM_CONFIG:-'Não definido'}"
    echo "EM_CACHE: ${EM_CACHE:-'Não definido'}"
    echo "PATH: $(echo $PATH | tr ':' '\n' | grep -E 'emsdk|emscripten' || echo '  Nenhum caminho Emscripten encontrado')"
    echo ""
    
    if command -v emcc &> /dev/null; then
        echo "Versões instaladas:"
        emcc --version | head -n 3
    fi
}

# Função principal
main() {
    local emsdk_path="${1:-/home/marcos/emsdk}"
    
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     LOAD EMSCRIPTEN ENVIRONMENT FOR CURRENT SHELL         ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    
    # Carregar o ambiente
    if load_emscripten_from_path "$emsdk_path"; then
        echo ""
        verify_emscripten
        echo ""
        test_compilation
        echo ""
        show_environment
        echo ""
        print_message "✅ Ambiente Emscripten pronto para uso!"
        echo ""
        print_message "Para usar, execute comandos como:"
        echo "  emcc hello.c -o hello.html"
        echo "  em++ hello.cpp -o hello.html"
    else
        print_error "❌ Falha ao carregar o ambiente Emscripten."
        print_message "Tente executar manualmente:"
        echo "  source /home/marcos/emsdk/emsdk_env.sh"
        exit 1
    fi
}

# Executar a função principal
main "$@"