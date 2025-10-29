
# Planejamento de Movimento em Braço Robótico 2-DOF
Implementação de algoritmos de busca em grafos (Dijkstra e A*) para planejamento de trajetória de um manipulador planar de dois elos, com detecção de colisões e diferentes funções de custo.
*Este trabalho foi feito com auxílio de ferramentas de IA.

**Autoras**: Carolina Ferrari e Debora Baldiotti  
**Disciplina**: Introdução à Inteligência Artificial  
**Professor**: Ricardo Poley  
**Período**: Outubro de 2025

## Índice
- [Descrição do Projeto](#-descrição-do-projeto)
- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Como Executar](#-como-executar)
- [Arquivos Gerados](#-arquivos-gerados)
- [Estrutura do Código](#-estrutura-do-código)
- [Parâmetros Configuráveis](#-parâmetros-configuráveis)
- [Resultados Esperados](#-resultados-esperados)
- [Documentação Completa](#-documentação-completa)

## Descrição do Projeto
Este projeto implementa um sistema de planejamento de movimento para um braço robótico planar de 2 graus de liberdade (2-DOF), utilizando técnicas de busca em grafos para encontrar trajetórias ótimas que evitam colisões com obstáculos.

### Características Principais
**Algoritmos de Busca**:
- Dijkstra (ótimo garantido)
- A* com heurística Manhattan (h1)
- A* com heurística Euclidiana (h2)

**Funções de Custo**:
- `c1`: Distância geométrica no C-space
- `c2`: Parcimônia com penalidade por proximidade a obstáculos
- `c3`: Estimativa de energia elétrica consumida

**Visualizações**:
- Mapas do C-space (espaço de configurações)
- Animações do braço no workspace
- Gráficos comparativos de desempenho

**Experimentos**:
- Comparação entre discretizações (Δθ = 5° e 2°)
- Análise de eficiência computacional
- Métricas de qualidade de trajetória

## Requisitos

### Sistema Operacional
- **Windows**: 10/11
- **Linux**: Ubuntu 20.04+ ou similar
- **macOS**: 10.15+ (Catalina ou superior)

### Python
- **Versão mínima**: Python 3.9
- **Versão recomendada**: Python 3.11 ou 3.13

### Bibliotecas Python
As seguintes bibliotecas são necessárias:
numpy>=1.24.0
matplotlib>=3.7.0

## Instalação

### Passo 1: Clone ou Baixe o Projeto
```bash
# Se estiver usando Git
git clone <url-do-repositorio>
cd T1

# Ou baixe e extraia o arquivo ZIP
```
### Passo 2: Crie um Ambiente Virtual (Recomendado)

#### Windows (PowerShell)
```powershell
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
.\venv\Scripts\Activate.ps1

# Se houver erro de política de execução, execute:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
#### Windows (CMD)
```cmd
python -m venv venv
venv\Scripts\activate.bat
```
#### Linux/macOS
```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate
```
### Passo 3: Instale as Dependências
Com o ambiente virtual ativado:

```bash
pip install --upgrade pip
pip install numpy matplotlib
```
**Ou usando requirements.txt**:

```bash
pip install -r requirements.txt
```

### Verificar Instalação
```bash
python -c "import numpy; import matplotlib; print('Instalação OK!')"
```

Se aparecer "Instalação OK!", você está pronto!

---

## Como Executar

### Execução Básica
Com o ambiente virtual ativado, execute:
```bash
python main.py
```

### O Que Acontece
O script irá:

1. **Construir o C-space** (espaço de configurações)
   - Gerar grade de discretização
   - Verificar colisões
   - Criar grafo de vizinhança
2. **Executar 6 experimentos**:
   - Exp 1: Dijkstra + c1 + Δθ=5°
   - Exp 2: A* (h1) + c1 + Δθ=5°
   - Exp 3: A* (h2) + c1 + Δθ=5°
   - Exp 4: Dijkstra + c2 + Δθ=5°
   - Exp 5: A* (h1) + c1 + Δθ=2°
   - Exp 6: Dijkstra + c3 + Δθ=5°
3. **Gerar visualizações**:
   - Mapas do C-space (PNG)
   - Frames do workspace (PNG)
   - Animação (GIF)
   - Gráficos comparativos (PNG)
4. **Imprimir resultados**:
   - Tabela comparativa no terminal
   - Custo, nós expandidos, tempo de execução

### Tempo de Execução Esperado
- **Total**: ~30-40 segundos
  - Construção dos grafos: ~15 segundos
  - Execução das buscas: <1 segundo
  - Geração de visualizações: ~10 segundos
  - Criação da animação GIF: ~10 segundos

## Arquivos Gerados
Após a execução, os seguintes arquivos serão criados no diretório atual:

### Visualizações do C-Space (Espaço de Configurações)
`cspace_dijkstra_cost1_5deg.png` | C-space com Dijkstra, custo geométrico |
`cspace_dijkstra_cost2_5deg.png` | C-space com Dijkstra, custo clearance |
`cspace_dijkstra_cost3_5deg.png` | C-space com Dijkstra, custo energia |
`cspace_astar_cost1_2deg.png` | C-space com A* (h1), discretização fina |

### Visualizações do Workspace (Espaço de Trabalho)
`workspace_dijkstra_cost1_5deg.png` | 12 frames da trajetória no workspace |
`animation_astar_h1.gif` | Animação completa da trajetória |

### Análises Comparativas
`comparison_plots.png` | 4 gráficos comparando todos os experimentos |

### Exemplo de Estrutura Após Execução
T1/
├── main.py
├── README.md
├── RELATORIO_TECNICO.md
├── requirements.txt
├── cspace_dijkstra_cost1_5deg.png
├── cspace_dijkstra_cost2_5deg.png
├── cspace_dijkstra_cost3_5deg.png
├── cspace_astar_cost1_2deg.png
├── workspace_dijkstra_cost1_5deg.png
├── animation_astar_h1.gif
└── comparison_plots.png

---

## Estrutura do Código

### Principais Classes
```python
# Parâmetros do robô
class RobotParameters:
    L1, L2                    # Comprimentos dos elos
    theta1_min, theta1_max    # Limites da junta 1
    theta2_min, theta2_max    # Limites da junta 2
    delta_theta               # Discretização angular

# Definição do problema
class ProblemInstance:
    theta_start               # Configuração inicial
    target_position           # Alvo no workspace
    obstacles                 # Lista de obstáculos retangulares

# Cinemática e física
class RobotArm:
    forward_kinematics()      # Calcula posições dos elos
    compute_gravity_torques() # Torques gravitacionais

# Detecção de colisões
class CollisionChecker:
    check_configuration()     # Verifica colisão de um nó
    check_edge()              # Verifica colisão ao longo de aresta

# Funções de custo
class CostFunctions:
    cost1_geometric()         # Distância geométrica
    cost2_parsimony_clearance() # Parcimônia + clearance
    cost3_energy()            # Energia elétrica

# Construção do grafo
class CSpaceGraph:
    discretize_joint_space()  # Gera grade de configurações
    build_graph()             # Constrói grafo de vizinhança
    visualize_cspace()        # Plota C-space

# Algoritmos de busca
class SearchAlgorithms:
    dijkstra()                # Algoritmo de Dijkstra
    a_star()                  # Algoritmo A*

# Visualização
class Visualizer:
    visualize_path_workspace() # Plota trajetória no workspace
    animate_path()            # Cria animação GIF
```

---

## Parâmetros Configuráveis

### Modificar Parâmetros do Robô
Edite a classe `RobotParameters` em `main.py`:

```python
class RobotParameters:
    # Geometria dos elos (metros)
    L1: float = 1.0
    L2: float = 0.7
    
    # Limites das juntas (graus)
    theta1_min: float = -90.0
    theta1_max: float = 90.0
    theta2_min: float = 0.0
    theta2_max: float = 150.0
    
    # Discretização (graus)
    delta_theta: float = 5.0  # Altere para 2.0 para maior precisão
```

### Modificar Obstáculos
Edite a classe `ProblemInstance`:

```python
class ProblemInstance:
    # Obstáculos [x_min, x_max, y_min, y_max]
    obstacles: List[List[float]] = field(default_factory=lambda: [
        [0.60, 1.00, 0.10, 0.50],  # Obstáculo 1
        [0.10, 0.35, -0.10, 0.10]   # Obstáculo 2
    ])
```

### Modificar Estado Inicial e Objetivo
```python
class ProblemInstance:
    # Estado inicial (graus)
    theta_start: Tuple[float, float] = (-40.0, 60.0)
    
    # Alvo no workspace (metros)
    target_position: Tuple[float, float] = (1.2, 0.3)
```

### Ajustar Funções de Custo
Nos experimentos (função `main()`):

```python
# Custo c2: Ajustar α e β
result4 = runner.run_experiment(
    delta_theta=5.0,
    cost_function='cost2',
    cost_params={'alpha': 1.0, 'beta': 0.02},  # Altere aqui
    algorithm='dijkstra'
)
```

## Resultados Esperados

### Saída no Terminal
```
================================================================================
PLANEJAMENTO DE MOVIMENTO EM BRAÇO ROBÓTICO 2-DOF
Algoritmos: Dijkstra e A*
================================================================================

================================================================================
EXPERIMENTO 1: Dijkstra + Custo Geométrico (c1) + Δθ=5°
================================================================================
Construindo o grafo no C-space...
Total de configurações na grade: 1147
Nós livres: 1109
Nós em colisão: 38
Nós goal encontrados: 2
Construindo arestas...
Arestas válidas criadas: 8392
Grafo construído com sucesso!

Executando Dijkstra...
✓ Caminho encontrado!
  Custo: 33.2843
  Nós expandidos: 122
  Tempo: 0.0003s

Mapa do C-space salvo em: cspace_dijkstra_cost1_5deg.png
Frames do workspace salvos em: workspace_dijkstra_cost1_5deg.png

[... outros experimentos ...]

================================================================================
RESUMO DOS EXPERIMENTOS
================================================================================
Δθ     Custo    Algoritmo  Heurística   Custo        Nós Exp.   Tempo (s)    Path Len
----------------------------------------------------------------------------------------
5.0    cost1    dijkstra   N/A          33.2843      122        0.0003       6
5.0    cost1    astar      h1           33.2843      6          0.0001       6
5.0    cost1    astar      h2           33.2843      10         0.0001       6
5.0    cost2    dijkstra   N/A          45.2774      171        0.0004       6
2.0    cost1    astar      h1           33.1127      13         0.0001       13
5.0    cost3    dijkstra   N/A          553047.6286  313        0.0007       6
========================================================================================

Gráficos comparativos salvos em: comparison_plots.png

================================================================================
EXPERIMENTOS CONCLUÍDOS!
================================================================================

Arquivos gerados:
  • Mapas do C-space: cspace_*.png
  • Frames do workspace: workspace_*.png
  • Gráficos comparativos: comparison_plots.png
  • Animação: animation_astar_h1.gif
```

## Documentação Completa
Para análise detalhada dos resultados, algoritmos e experimentos, consulte o relatório do trabalho.

## Créditos
**Autoras**:
- Carolina Ferrari
- Debora Baldiotti

**Disciplina**: Introdução à Inteligência Artificial  
**Instituição**: Universidade Federal de Minas Gerais
**Professor**: Ricardo Poley  
**Período**: Outubro de 2025  

**Referências**:

Disponíveis no relatório.

---

**Última atualização**: 28 de outubro de 2025