import sys
import io

# configura stdout para utf-8 no windows para evitar unicodeencodeerror
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional
import time
from collections import defaultdict

# parametros da instancia base

@dataclass
class RobotParameters:
    """Parâmetros geométricos e físicos do robô"""
    # geometria dos elos
    L1: float = 1.0  # comprimento do elo 1 (m)
    L2: float = 0.7  # comprimento do elo 2 (m)
    
    # limites das juntas (em graus)
    theta1_min: float = -90.0
    theta1_max: float = 90.0
    theta2_min: float = 0.0
    theta2_max: float = 150.0
    
    # discretizacao
    delta_theta: float = 5.0  # passo de discretizacao (graus)
    
    # parametros fisicos para custo de energia
    m1: float = 3.0  # massa do elo 1 (kg)
    m2: float = 2.0  # massa do elo 2 (kg)
    l1c: float = 0.5  # centro de massa do elo 1 (m)
    l2c: float = 0.35  # centro de massa do elo 2 (m)
    I1: float = 0.4  # inercia do elo 1 (kg m2)
    I2: float = 0.2  # inercia do elo 2 (kg m2)
    
    # atritos
    b1: float = 0.05  # atrito viscoso junta 1 (nms/rad)
    b2: float = 0.05  # atrito viscoso junta 2 (nms/rad)
    tau_c1: float = 0.3  # atrito de coulomb junta 1 (nm)
    tau_c2: float = 0.3  # atrito de coulomb junta 2 (nm)
    
    # motores/eletronica
    Kt: float = 0.08  # constante de torque (nm/a)
    R: float = 1.2  # resistencia (ohm)
    P_idle: float = 6.0  # potencia idle (w)
    
    # perfil de movimento
    omega_max: float = 60.0  # velocidade maxima (graus/s)
    alpha_max: float = 120.0  # aceleracao maxima (graus/s2)
    
    # tolerancias
    epsilon: float = 0.001  # tolerancia geometrica (m)
    goal_tolerance: float = 0.05  # tolerancia para alcancar o alvo (m)


@dataclass
class ProblemInstance:
    """definicao da instancia do problema"""
    # estado inicial (graus)
    theta_start: Tuple[float, float] = (-40.0, 60.0)
    
    # alvo no espaco de trabalho (m)
    target_position: Tuple[float, float] = (1.2, 0.3)
    
    # obstaculos retangulares [x_min, x_max, y_min, y_max]
    obstacles: List[List[float]] = field(default_factory=lambda: [
    [0.60, 1.00, 0.10, 0.50],  # o1
    [0.10, 0.35, -0.10, 0.10]   # o2
    ])


# cinematica direta e geometrica

class RobotArm:
    """classe para o braco robotico planar 2-dof"""
    
    def __init__(self, params: RobotParameters):
        self.params = params
        self.g = 9.81  # aceleracao da gravidade (m/s2)
    
    def forward_kinematics(self, theta1: float, theta2: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula as posições das juntas P1 e P2 (efetuador final)
        
        Args:
            theta1, theta2: Ângulos das juntas em graus
            
        Returns:
            P1, P2: Posições das juntas como arrays numpy [x, y]
        """
        # Converter para radianos
        th1_rad = np.radians(theta1)
        th2_rad = np.radians(theta2)
        
        # Posição da junta 1
        P1 = np.array([
            self.params.L1 * np.cos(th1_rad),
            self.params.L1 * np.sin(th1_rad)
        ])
        
        # Posição do efetuador final
        P2 = P1 + np.array([
            self.params.L2 * np.cos(th1_rad + th2_rad),
            self.params.L2 * np.sin(th1_rad + th2_rad)
        ])
        
        return P1, P2
    
    def get_link_segments(self, theta1: float, theta2: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna os segmentos dos elos
        
        Returns:
            seg1, seg2: Arrays de formato (2, 2) com [[x_start, y_start], [x_end, y_end]]
        """
        P1, P2 = self.forward_kinematics(theta1, theta2)
        base = np.array([0.0, 0.0])
        
        seg1 = np.array([base, P1])
        seg2 = np.array([P1, P2])
        
        return seg1, seg2
    
    def compute_gravity_torques(self, theta1: float, theta2: float) -> Tuple[float, float]:
        """
        Calcula os torques gravitacionais (modelo quasiestático)
        
        Returns:
            g1, g2: Torques gravitacionais nas juntas 1 e 2 (N·m)
        """
        th1_rad = np.radians(theta1)
        th12_rad = np.radians(theta1 + theta2)
        
        g1 = (self.params.m1 * self.params.l1c + 
              self.params.m2 * self.params.L1) * self.g * np.cos(th1_rad) + \
             self.params.m2 * self.params.l2c * self.g * np.cos(th12_rad)
        
        g2 = self.params.m2 * self.params.l2c * self.g * np.cos(th12_rad)
        
        return g1, g2


# =============================================================================
# DETECÇÃO DE COLISÕES
# =============================================================================

class CollisionChecker:
    """Verifica colisões entre segmentos de reta e obstáculos retangulares"""
    
    def __init__(self, obstacles: List[List[float]], epsilon: float = 0.001):
        """
        Args:
            obstacles: Lista de obstáculos [x_min, x_max, y_min, y_max]
            epsilon: Margem de segurança (m)
        """
        self.obstacles = obstacles
        self.epsilon = epsilon
    
    def point_in_rectangle(self, point: np.ndarray, rect: List[float]) -> bool:
        """Verifica se um ponto está dentro de um retângulo (com margem epsilon)"""
        x, y = point
        x_min, x_max, y_min, y_max = rect
        return (x_min - self.epsilon <= x <= x_max + self.epsilon and
                y_min - self.epsilon <= y <= y_max + self.epsilon)
    
    def segment_rectangle_intersection(self, seg: np.ndarray, rect: List[float]) -> bool:
        """
        Verifica se um segmento de reta intersecta um retângulo
        
        Args:
            seg: Array (2, 2) com [[x1, y1], [x2, y2]]
            rect: [x_min, x_max, y_min, y_max]
            
        Returns:
            True se houver interseção
        """
        p1, p2 = seg[0], seg[1]
        
        # Verifica se os extremos estão dentro do retângulo
        if self.point_in_rectangle(p1, rect) or self.point_in_rectangle(p2, rect):
            return True
        
        # Verifica interseção com as arestas do retângulo
        x_min, x_max, y_min, y_max = rect
        
        # Expande o retângulo com epsilon
        x_min -= self.epsilon
        x_max += self.epsilon
        y_min -= self.epsilon
        y_max += self.epsilon
        
        # Define as 4 arestas do retângulo
        rect_edges = [
            np.array([[x_min, y_min], [x_max, y_min]]),  # Bottom
            np.array([[x_max, y_min], [x_max, y_max]]),  # Right
            np.array([[x_max, y_max], [x_min, y_max]]),  # Top
            np.array([[x_min, y_max], [x_min, y_min]])   # Left
        ]
        
        # Verifica interseção com cada aresta
        for edge in rect_edges:
            if self.segments_intersect(seg, edge):
                return True
        
        return False
    
    def segments_intersect(self, seg1: np.ndarray, seg2: np.ndarray) -> bool:
        """
        Verifica se dois segmentos de reta se intersectam
        Usa o método de orientação e pontos colineares
        """
        def ccw(A, B, C):
            """Retorna True se A-B-C faz uma curva anti-horária"""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        A, B = seg1[0], seg1[1]
        C, D = seg2[0], seg2[1]
        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    def check_configuration(self, robot: RobotArm, theta1: float, theta2: float) -> bool:
        """
        Verifica se uma configuração do robô está livre de colisões
        
        IMPORTANTE: Esta implementação verifica colisão APENAS da ponta do braço (P2).
        Os elos intermediários NÃO são verificados para colisão.
        
        Justificativa: Simplificação do problema conforme especificações do trabalho.
        Se for necessário verificar colisão dos elos, modifique este método para usar
        segment_rectangle_intersection() nos segmentos [Base->P1] e [P1->P2].
        
        Returns:
            True se livre de colisões, False se em colisão
        """
        # Obtém apenas a posição da ponta (efetuador final)
        _, P2 = robot.forward_kinematics(theta1, theta2)
        
        # Verifica se a ponta está em colisão com algum obstáculo
        for obstacle in self.obstacles:
            if self.point_in_rectangle(P2, obstacle):
                return False
        
        return True
    
    def check_edge(self, robot: RobotArm, theta_from: Tuple[float, float], 
                   theta_to: Tuple[float, float], num_samples: int = 5) -> bool:
        """
        Verifica se uma aresta (transição) está livre de colisões
        Interpola linearmente entre as configurações e verifica colisões
        
        IMPORTANTE: Esta implementação verifica colisão APENAS da ponta do braço (P2).
        
        Returns:
            True se livre de colisões, False se houver colisão
        """
        for i in range(num_samples + 1):
            t = i / num_samples
            theta1 = (1 - t) * theta_from[0] + t * theta_to[0]
            theta2 = (1 - t) * theta_from[1] + t * theta_to[1]
            
            if not self.check_configuration(robot, theta1, theta2):
                return False
        
        return True
    
    def min_distance_to_obstacles(self, robot: RobotArm, theta1: float, theta2: float) -> float:
        """
        Calcula a distância mínima da ponta do braço (P2) aos obstáculos
        
        IMPORTANTE: Considera apenas a ponta (P2), não os elos intermediários.
        
        Returns:
            Distância mínima em metros
        """
        _, P2 = robot.forward_kinematics(theta1, theta2)
        min_dist = float('inf')
        
        for obstacle in self.obstacles:
            dist = self.point_to_rectangle_distance(P2, obstacle)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def point_to_segment_distance(self, point: np.ndarray, seg: np.ndarray) -> float:
        """Calcula a distância de um ponto a um segmento de reta"""
        p1, p2 = seg[0], seg[1]
        line_vec = p2 - p1
        point_vec = point - p1
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-10:
            return np.linalg.norm(point_vec)
        
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        
        nearest = p1 + t * line_vec
        return np.linalg.norm(point - nearest)
    
    def point_to_rectangle_distance(self, point: np.ndarray, rect: List[float]) -> float:
        """Calcula a distância de um ponto a um retângulo"""
        x, y = point
        x_min, x_max, y_min, y_max = rect
        
        dx = max(x_min - x, 0, x - x_max)
        dy = max(y_min - y, 0, y - y_max)
        
        return np.sqrt(dx**2 + dy**2)


# =============================================================================
# FUNÇÕES DE CUSTO
# =============================================================================

class CostFunctions:
    """Implementa diferentes funções de custo para as arestas"""
    
    def __init__(self, robot: RobotArm, collision_checker: CollisionChecker):
        self.robot = robot
        self.collision_checker = collision_checker
    
    def cost1_geometric(self, theta_from: Tuple[float, float], 
                        theta_to: Tuple[float, float],
                        w1: float = 1.0, w2: float = 1.0) -> float:
        """
        Custo 1: Distância geométrica no espaço de juntas
        c1 = sqrt(w1*(Δθ1)² + w2*(Δθ2)²)
        """
        delta_theta1 = theta_to[0] - theta_from[0]
        delta_theta2 = theta_to[1] - theta_from[1]
        
        return np.sqrt(w1 * delta_theta1**2 + w2 * delta_theta2**2)
    
    def cost2_parsimony_clearance(self, theta_from: Tuple[float, float],
                                   theta_to: Tuple[float, float],
                                   alpha: float = 1.0, beta: float = 0.02) -> float:
        """
        Custo 2: Parcimônia + penalidade por proximidade a obstáculos
        c2 = α(|Δθ1| + |Δθ2|) + β/(d + ε)
        """
        delta_theta1 = abs(theta_to[0] - theta_from[0])
        delta_theta2 = abs(theta_to[1] - theta_from[1])
        
        parsimony = alpha * (delta_theta1 + delta_theta2)
        
        # Calcula distância mínima ao longo da aresta (amostra 3 pontos)
        min_clearance = float('inf')
        for t in [0.0, 0.5, 1.0]:
            theta1 = (1 - t) * theta_from[0] + t * theta_to[0]
            theta2 = (1 - t) * theta_from[1] + t * theta_to[1]
            dist = self.collision_checker.min_distance_to_obstacles(self.robot, theta1, theta2)
            min_clearance = min(min_clearance, dist)
        
        proximity_penalty = beta / (min_clearance + self.robot.params.epsilon)
        
        return parsimony + proximity_penalty
    
    def cost3_energy(self, theta_from: Tuple[float, float],
                     theta_to: Tuple[float, float],
                     num_samples: int = 5) -> float:
        """
        Custo 3: Aproximação da energia elétrica consumida
        Usa modelo quasiestático com perfil trapezoidal de movimento
        """
        delta_theta1 = theta_to[0] - theta_from[0]
        delta_theta2 = theta_to[1] - theta_from[1]
        
        # Tempo estimado para o movimento (perfil trapezoidal simplificado)
        max_delta = max(abs(delta_theta1), abs(delta_theta2))
        
        # Converte para rad/s e rad/s²
        omega_max_rad = np.radians(self.robot.params.omega_max)
        alpha_max_rad = np.radians(self.robot.params.alpha_max)
        
        # Tempo de aceleração
        t_accel = omega_max_rad / alpha_max_rad
        
        # Distância durante aceleração e desaceleração
        dist_accel = 0.5 * alpha_max_rad * t_accel**2
        dist_total = np.radians(max_delta)
        
        if 2 * dist_accel >= dist_total:
            # Movimento triangular (não atinge velocidade máxima)
            t_total = 2 * np.sqrt(dist_total / alpha_max_rad)
        else:
            # Movimento trapezoidal
            t_cruise = (dist_total - 2 * dist_accel) / omega_max_rad
            t_total = 2 * t_accel + t_cruise
        
        # Amostra o caminho e calcula energia
        energy = 0.0
        dt = t_total / num_samples
        
        for i in range(num_samples + 1):
            t = i / num_samples
            theta1 = (1 - t) * theta_from[0] + t * theta_to[0]
            theta2 = (1 - t) * theta_from[1] + t * theta_to[1]
            
            # Velocidades angulares estimadas (simplificado)
            omega1 = delta_theta1 / (t_total + 1e-10)  # graus/s
            omega2 = delta_theta2 / (t_total + 1e-10)
            
            # Torques gravitacionais
            g1, g2 = self.robot.compute_gravity_torques(theta1, theta2)
            
            # Torques de atrito
            omega1_rad = np.radians(omega1)
            omega2_rad = np.radians(omega2)
            
            tau1 = g1 + self.robot.params.b1 * omega1_rad + \
                   self.robot.params.tau_c1 * np.sign(omega1_rad)
            tau2 = g2 + self.robot.params.b2 * omega2_rad + \
                   self.robot.params.tau_c2 * np.sign(omega2_rad)
            
            # Energia elétrica (simplificado)
            Kt = self.robot.params.Kt
            R = self.robot.params.R
            
            # Potência mecânica
            P_mech = abs(tau1 * omega1_rad) + abs(tau2 * omega2_rad)
            
            # Perdas resistivas (I²R)
            I1 = tau1 / Kt
            I2 = tau2 / Kt
            P_loss = R * (I1**2 + I2**2)
            
            # Energia total neste intervalo
            energy += (P_mech + P_loss + self.robot.params.P_idle) * dt
        
        return energy


# =============================================================================
# CONSTRUÇÃO DO GRAFO NO C-SPACE
# =============================================================================

class CSpaceGraph:
    """Constrói e gerencia o grafo no espaço de configurações"""
    
    def __init__(self, robot: RobotArm, collision_checker: CollisionChecker,
                 problem: ProblemInstance):
        self.robot = robot
        self.collision_checker = collision_checker
        self.problem = problem
        self.params = robot.params
        
        # Nós do grafo: dict de (theta1, theta2) -> node_id
        self.nodes: Dict[Tuple[float, float], int] = {}
        self.node_ids: Dict[int, Tuple[float, float]] = {}
        self.node_counter = 0
        
        # Arestas: dict de node_id -> [(neighbor_id, cost), ...]
        self.edges: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        
        # Nós goal (múltiplos nós podem satisfazer a condição de goal)
        self.goal_nodes: Set[int] = set()
        
        # Estatísticas
        self.free_nodes = 0
        self.collision_nodes = 0
        
    def discretize_joint_space(self) -> List[Tuple[float, float]]:
        """
        Gera a grade de discretização do espaço de juntas
        
        Returns:
            Lista de configurações (theta1, theta2) em graus
        """
        theta1_range = np.arange(
            self.params.theta1_min,
            self.params.theta1_max + self.params.delta_theta,
            self.params.delta_theta
        )
        
        theta2_range = np.arange(
            self.params.theta2_min,
            self.params.theta2_max + self.params.delta_theta,
            self.params.delta_theta
        )
        
        configurations = []
        for theta1 in theta1_range:
            for theta2 in theta2_range:
                configurations.append((float(theta1), float(theta2)))
        
        return configurations
    
    def build_graph(self, cost_function_name: str = 'cost1', 
                    cost_params: dict = None) -> None:
        """
        Constrói o grafo completo: nós livres de colisão e arestas válidas
        
        Args:
            cost_function_name: Nome da função de custo ('cost1', 'cost2', 'cost3')
            cost_params: Parâmetros adicionais para a função de custo
        """
        print("Construindo o grafo no C-space...")
        
        if cost_params is None:
            cost_params = {}
        
        # Cria objeto de funções de custo
        cost_funcs = CostFunctions(self.robot, self.collision_checker)
        
        # Seleciona a função de custo
        if cost_function_name == 'cost1':
            cost_func = cost_funcs.cost1_geometric
        elif cost_function_name == 'cost2':
            cost_func = lambda f, t: cost_funcs.cost2_parsimony_clearance(
                f, t, **cost_params)
        elif cost_function_name == 'cost3':
            cost_func = cost_funcs.cost3_energy
        else:
            raise ValueError(f"Função de custo desconhecida: {cost_function_name}")
        
        # Gera todas as configurações
        configurations = self.discretize_joint_space()
        print(f"Total de configurações na grade: {len(configurations)}")
        
        # Adiciona nós livres de colisão
        for config in configurations:
            if self.collision_checker.check_configuration(self.robot, config[0], config[1]):
                node_id = self.node_counter
                self.nodes[config] = node_id
                self.node_ids[node_id] = config
                self.node_counter += 1
                self.free_nodes += 1
                
                # Verifica se é nó goal
                if self.is_goal_node(config):
                    self.goal_nodes.add(node_id)
            else:
                self.collision_nodes += 1
        
        print(f"Nós livres: {self.free_nodes}")
        print(f"Nós em colisão: {self.collision_nodes}")
        print(f"Nós goal encontrados: {len(self.goal_nodes)}")
        
        # Adiciona arestas (8-conectividade)
        print("Construindo arestas...")
        delta = self.params.delta_theta
        movements = [
            (-delta, -delta), (-delta, 0), (-delta, delta),
            (0, -delta),                   (0, delta),
            (delta, -delta),  (delta, 0),  (delta, delta)
        ]
        
        edge_count = 0
        for config, node_id in self.nodes.items():
            for d_theta1, d_theta2 in movements:
                neighbor_config = (config[0] + d_theta1, config[1] + d_theta2)
                
                # Verifica se o vizinho existe e está nos limites
                if neighbor_config in self.nodes:
                    neighbor_id = self.nodes[neighbor_config]
                    
                    # Verifica se a aresta está livre de colisões
                    if self.collision_checker.check_edge(self.robot, config, neighbor_config):
                        # Calcula o custo da aresta
                        cost = cost_func(config, neighbor_config)
                        self.edges[node_id].append((neighbor_id, cost))
                        edge_count += 1
        
        print(f"Arestas válidas criadas: {edge_count}")
        print("Grafo construído com sucesso!\n")
    
    def is_goal_node(self, config: Tuple[float, float]) -> bool:
        """
        Verifica se uma configuração atinge o alvo no espaço de trabalho
        
        Returns:
            True se a distância do efetuador ao alvo é <= goal_tolerance
        """
        _, P2 = self.robot.forward_kinematics(config[0], config[1])
        target = np.array(self.problem.target_position)
        distance = np.linalg.norm(P2 - target)
        
        return distance <= self.params.goal_tolerance
    
    def get_start_node(self) -> Optional[int]:
        """Retorna o ID do nó inicial"""
        start_config = self.problem.theta_start
        return self.nodes.get(start_config)
    
    def visualize_cspace(self, path_nodes: List[int] = None, 
                         save_path: str = None, 
                         algorithm_info: str = None) -> None:
        """
        Visualiza o C-space com nós livres, ocupados e o caminho encontrado
        
        Args:
            path_nodes: Lista de IDs dos nós do caminho (opcional)
            save_path: Caminho para salvar a figura (opcional)
            algorithm_info: Informação sobre algoritmo e parâmetros (opcional)
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plota todos os pontos da grade
        all_configs = self.discretize_joint_space()
        
        free_configs = [c for c in all_configs if c in self.nodes]
        collision_configs = [c for c in all_configs if c not in self.nodes]
        
        # Nós em colisão
        if collision_configs:
            collision_array = np.array(collision_configs)
            ax.scatter(collision_array[:, 0], collision_array[:, 1],
                      c='red', s=20, alpha=0.3, label='Colisão', marker='s')
        
        # Nós livres
        if free_configs:
            free_array = np.array(free_configs)
            ax.scatter(free_array[:, 0], free_array[:, 1],
                      c='lightblue', s=20, alpha=0.5, label='Livre', marker='o')
        
        # Nós goal
        if self.goal_nodes:
            goal_configs = [self.node_ids[gid] for gid in self.goal_nodes]
            goal_array = np.array(goal_configs)
            ax.scatter(goal_array[:, 0], goal_array[:, 1],
                      c='gold', s=100, marker='*', 
                      edgecolors='orange', linewidths=2,
                      label='Goal', zorder=5)
        
        # Nó inicial
        start_config = self.problem.theta_start
        ax.scatter(start_config[0], start_config[1],
                  c='green', s=200, marker='o',
                  edgecolors='darkgreen', linewidths=2,
                  label='Start', zorder=5)
        
        # Caminho encontrado
        if path_nodes and len(path_nodes) > 1:
            path_configs = [self.node_ids[nid] for nid in path_nodes]
            path_array = np.array(path_configs)
            ax.plot(path_array[:, 0], path_array[:, 1],
                   'b-', linewidth=2.5, label='Caminho', zorder=4)
            ax.scatter(path_array[:, 0], path_array[:, 1],
                      c='blue', s=50, zorder=4)
        
        ax.set_xlabel('θ₁ - Ângulo da Junta 1 (graus)', fontsize=14)
        ax.set_ylabel('θ₂ - Ângulo da Junta 2 (graus)', fontsize=14)
        
        # Título com informações do algoritmo
        if algorithm_info:
            title = f'Espaço de Configurações (C-space)\n{algorithm_info}'
        else:
            title = 'Espaço de Configurações (C-space)'
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([self.params.theta1_min - 5, self.params.theta1_max + 5])
        ax.set_ylim([self.params.theta2_min - 5, self.params.theta2_max + 5])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Mapa do C-space salvo em: {save_path}")
        
        plt.show()


# =============================================================================
# ALGORITMOS DE BUSCA
# =============================================================================

@dataclass(order=True)
class PriorityNode:
    """Nó para a fila de prioridade"""
    priority: float
    node_id: int = field(compare=False)


class SearchAlgorithms:
    """Implementa algoritmos de busca Dijkstra e A*"""
    
    def __init__(self, graph: CSpaceGraph):
        self.graph = graph
        
        # Estatísticas da busca
        self.nodes_expanded = 0
        self.search_time = 0.0
        self.path_cost = 0.0
    
    def dijkstra(self, start_id: int) -> Tuple[Optional[List[int]], Dict[int, float]]:
        """
        Algoritmo de Dijkstra
        
        Args:
            start_id: ID do nó inicial
            
        Returns:
            path: Lista de IDs dos nós do caminho (None se não encontrado)
            distances: Dicionário com distâncias mínimas de cada nó
        """
        print("Executando Dijkstra...")
        start_time = time.time()
        
        # Verifica se existem nós goal
        if not self.graph.goal_nodes:
            print("AVISO: Nenhum nó goal encontrado! Buscando nó mais próximo ao alvo...")
            # Encontra o nó mais próximo ao alvo
            target = np.array(self.graph.problem.target_position)
            min_dist = float('inf')
            closest_node = None
            
            for node_id, config in self.graph.node_ids.items():
                _, P2 = self.graph.robot.forward_kinematics(config[0], config[1])
                dist = np.linalg.norm(P2 - target)
                if dist < min_dist:
                    min_dist = dist
                    closest_node = node_id
            
            if closest_node is not None:
                self.graph.goal_nodes.add(closest_node)
                goal_config = self.graph.node_ids[closest_node]
                print(f"  Nó mais próximo: θ1={goal_config[0]}°, θ2={goal_config[1]}°, dist={min_dist:.4f}m")
            else:
                self.search_time = time.time() - start_time
                print("✗ Nenhum caminho encontrado!\n")
                return None, {}
        
        # Inicialização
        distances = {start_id: 0.0}
        previous = {}
        visited = set()
        priority_queue = [PriorityNode(0.0, start_id)]
        self.nodes_expanded = 0
        
        while priority_queue:
            current = heapq.heappop(priority_queue)
            current_id = current.node_id
            current_dist = current.priority
            
            # Se já visitou, pula
            if current_id in visited:
                continue
            
            visited.add(current_id)
            self.nodes_expanded += 1
            
            # Verifica se chegou ao goal
            if current_id in self.graph.goal_nodes:
                self.search_time = time.time() - start_time
                self.path_cost = distances[current_id]
                path = self.reconstruct_path(previous, start_id, current_id)
                print(f"✓ Caminho encontrado!")
                print(f"  Custo: {self.path_cost:.4f}")
                print(f"  Nós expandidos: {self.nodes_expanded}")
                print(f"  Tempo: {self.search_time:.4f}s\n")
                return path, distances
            
            # Expande vizinhos
            for neighbor_id, edge_cost in self.graph.edges[current_id]:
                if neighbor_id in visited:
                    continue
                
                new_distance = current_dist + edge_cost
                
                if neighbor_id not in distances or new_distance < distances[neighbor_id]:
                    distances[neighbor_id] = new_distance
                    previous[neighbor_id] = current_id
                    heapq.heappush(priority_queue, 
                                 PriorityNode(new_distance, neighbor_id))
        
        self.search_time = time.time() - start_time
        print("✗ Nenhum caminho encontrado!\n")
        return None, distances
    
    def a_star(self, start_id: int, heuristic_name: str = 'h1') -> Tuple[Optional[List[int]], Dict[int, float]]:
        """
        Algoritmo A*
        
        Args:
            start_id: ID do nó inicial
            heuristic_name: Nome da heurística ('h1' ou 'h2')
            
        Returns:
            path: Lista de IDs dos nós do caminho (None se não encontrado)
            g_costs: Dicionário com custos g de cada nó
        """
        print(f"Executando A* com heurística {heuristic_name}...")
        start_time = time.time()
        
        # Verifica se existem nós goal
        if not self.graph.goal_nodes:
            print("AVISO: Nenhum nó goal encontrado! Buscando nó mais próximo ao alvo...")
            # Encontra o nó mais próximo ao alvo
            target = np.array(self.graph.problem.target_position)
            min_dist = float('inf')
            closest_node = None
            
            for node_id, config in self.graph.node_ids.items():
                _, P2 = self.graph.robot.forward_kinematics(config[0], config[1])
                dist = np.linalg.norm(P2 - target)
                if dist < min_dist:
                    min_dist = dist
                    closest_node = node_id
            
            if closest_node is not None:
                self.graph.goal_nodes.add(closest_node)
                goal_config = self.graph.node_ids[closest_node]
                print(f"  Nó mais próximo: θ1={goal_config[0]}°, θ2={goal_config[1]}°, dist={min_dist:.4f}m")
            else:
                self.search_time = time.time() - start_time
                print("✗ Nenhum caminho encontrado!\n")
                return None, {}
        
        # Seleciona um nó goal representativo para a heurística
        goal_id = next(iter(self.graph.goal_nodes))
        goal_config = self.graph.node_ids[goal_id]
        
        # Função heurística
        if heuristic_name == 'h1':
            def heuristic(node_id):
                config = self.graph.node_ids[node_id]
                return abs(config[0] - goal_config[0]) + abs(config[1] - goal_config[1])
        elif heuristic_name == 'h2':
            def heuristic(node_id):
                config = self.graph.node_ids[node_id]
                return np.sqrt((config[0] - goal_config[0])**2 + 
                             (config[1] - goal_config[1])**2)
        else:
            raise ValueError(f"Heurística desconhecida: {heuristic_name}")
        
        # Inicialização
        g_costs = {start_id: 0.0}
        f_costs = {start_id: heuristic(start_id)}
        previous = {}
        visited = set()
        priority_queue = [PriorityNode(f_costs[start_id], start_id)]
        self.nodes_expanded = 0
        
        while priority_queue:
            current = heapq.heappop(priority_queue)
            current_id = current.node_id
            
            # Se já visitou, pula
            if current_id in visited:
                continue
            
            visited.add(current_id)
            self.nodes_expanded += 1
            
            # Verifica se chegou ao goal
            if current_id in self.graph.goal_nodes:
                self.search_time = time.time() - start_time
                self.path_cost = g_costs[current_id]
                path = self.reconstruct_path(previous, start_id, current_id)
                print(f"✓ Caminho encontrado!")
                print(f"  Custo: {self.path_cost:.4f}")
                print(f"  Nós expandidos: {self.nodes_expanded}")
                print(f"  Tempo: {self.search_time:.4f}s\n")
                return path, g_costs
            
            # Expande vizinhos
            current_g = g_costs[current_id]
            
            for neighbor_id, edge_cost in self.graph.edges[current_id]:
                if neighbor_id in visited:
                    continue
                
                tentative_g = current_g + edge_cost
                
                if neighbor_id not in g_costs or tentative_g < g_costs[neighbor_id]:
                    g_costs[neighbor_id] = tentative_g
                    f_cost = tentative_g + heuristic(neighbor_id)
                    f_costs[neighbor_id] = f_cost
                    previous[neighbor_id] = current_id
                    heapq.heappush(priority_queue, 
                                 PriorityNode(f_cost, neighbor_id))
        
        self.search_time = time.time() - start_time
        print("✗ Nenhum caminho encontrado!\n")
        return None, g_costs
    
    def reconstruct_path(self, previous: Dict[int, int], 
                        start_id: int, goal_id: int) -> List[int]:
        """Reconstrói o caminho a partir do dicionário de predecessores"""
        path = [goal_id]
        current = goal_id
        
        while current != start_id:
            current = previous[current]
            path.append(current)
        
        path.reverse()
        return path


# =============================================================================
# VISUALIZAÇÃO E ANIMAÇÃO
# =============================================================================

class Visualizer:
    """Classe para visualização e animação do braço robótico"""
    
    def __init__(self, robot: RobotArm, problem: ProblemInstance):
        self.robot = robot
        self.problem = problem
    
    def plot_workspace_frame(self, theta1: float, theta2: float, 
                            ax: plt.Axes, show_target: bool = True) -> None:
        """
        Plota um frame do braço robótico no workspace
        
        Args:
            theta1, theta2: Ângulos das juntas em graus
            ax: Eixo matplotlib
            show_target: Se True, mostra o alvo
        """
        # Cinemática direta
        P1, P2 = self.robot.forward_kinematics(theta1, theta2)
        base = np.array([0.0, 0.0])
        
        # Plota os elos
        ax.plot([base[0], P1[0]], [base[1], P1[1]], 
               'b-', linewidth=6, label='Elo 1' if show_target else '')
        ax.plot([P1[0], P2[0]], [P1[1], P2[1]], 
               'r-', linewidth=6, label='Elo 2' if show_target else '')
        
        # Plota as juntas
        ax.plot(base[0], base[1], 'ko', markersize=12, label='Base')
        ax.plot(P1[0], P1[1], 'go', markersize=10, label='Junta 1')
        ax.plot(P2[0], P2[1], 'mo', markersize=12, label='Efetuador')
        
        # Plota obstáculos
        for i, obs in enumerate(self.problem.obstacles):
            x_min, x_max, y_min, y_max = obs
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height,
                                     linewidth=2, edgecolor='black',
                                     facecolor='gray', alpha=0.7,
                                     label=f'Obstáculo {i+1}' if show_target else '')
            ax.add_patch(rect)
        
        # Plota o alvo
        if show_target:
            target = self.problem.target_position
            ax.plot(target[0], target[1], 'y*', markersize=20,
                   markeredgecolor='orange', markeredgewidth=2, label='Alvo')
            
            # Círculo de tolerância
            circle = plt.Circle(target, self.robot.params.goal_tolerance,
                              color='yellow', fill=False, linestyle='--',
                              linewidth=2, alpha=0.5)
            ax.add_patch(circle)
    
    def visualize_path_workspace(self, path_nodes: List[int], 
                                 graph: CSpaceGraph,
                                 num_frames: int = 10,
                                 save_path: str = None,
                                 algorithm_info: str = None) -> None:
        """
        Visualiza múltiplos frames do caminho no workspace
        
        Args:
            path_nodes: Lista de IDs dos nós do caminho
            graph: Grafo com os nós
            num_frames: Número de frames a mostrar
            save_path: Caminho para salvar a figura (opcional)
            algorithm_info: Informação sobre algoritmo e parâmetros (opcional)
        """
        # Seleciona frames uniformemente espaçados
        indices = np.linspace(0, len(path_nodes) - 1, num_frames, dtype=int)
        
        # Calcula o layout da grade
        cols = int(np.ceil(np.sqrt(num_frames)))
        rows = int(np.ceil(num_frames / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes = axes.flatten() if num_frames > 1 else [axes]
        
        for idx, ax in enumerate(axes):
            if idx < len(indices):
                node_id = path_nodes[indices[idx]]
                config = graph.node_ids[node_id]
                
                self.plot_workspace_frame(config[0], config[1], ax, 
                                        show_target=(idx == 0))
                
                ax.set_xlim([-0.5, 2.0])
                ax.set_ylim([-1.0, 1.5])
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlabel('x - Posição no eixo X (m)', fontsize=10)
                ax.set_ylabel('y - Posição no eixo Y (m)', fontsize=10)
                ax.set_title(f'Frame {indices[idx]+1}/{len(path_nodes)}\n' +
                           f'Junta 1: θ₁={config[0]:.1f}°, Junta 2: θ₂={config[1]:.1f}°',
                           fontsize=10)
                
                if idx == 0:
                    ax.legend(fontsize=8, loc='upper left')
            else:
                ax.axis('off')
        
        # Título com informações do algoritmo
        if algorithm_info:
            title = f'Trajetória do Robô no Workspace (Espaço de Trabalho)\n{algorithm_info}'
        else:
            title = 'Trajetória do Robô no Workspace (Espaço de Trabalho)'
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Frames do workspace salvos em: {save_path}")
        
        plt.show()
    
    def animate_path(self, path_nodes: List[int], graph: CSpaceGraph,
                    save_path: str = None, interval: int = 100) -> None:
        """
        Cria uma animação do caminho no workspace
        
        Args:
            path_nodes: Lista de IDs dos nós do caminho
            graph: Grafo com os nós
            save_path: Caminho para salvar a animação (opcional)
            interval: Intervalo entre frames em ms
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Configuração inicial dos obstáculos e alvo (fixos)
        for i, obs in enumerate(self.problem.obstacles):
            x_min, x_max, y_min, y_max = obs
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height,
                                     linewidth=2, edgecolor='black',
                                     facecolor='gray', alpha=0.7)
            ax.add_patch(rect)
        
        target = self.problem.target_position
        ax.plot(target[0], target[1], 'y*', markersize=20,
               markeredgecolor='orange', markeredgewidth=2)
        circle = plt.Circle(target, self.robot.params.goal_tolerance,
                          color='yellow', fill=False, linestyle='--',
                          linewidth=2, alpha=0.5)
        ax.add_patch(circle)
        
        # Traço do efetuador
        end_effector_trace = [[], []]
        trace_line, = ax.plot([], [], 'c--', linewidth=1, alpha=0.5, label='Trajetória')
        
        # Elementos que serão atualizados
        link1_line, = ax.plot([], [], 'b-', linewidth=6, label='Elo 1')
        link2_line, = ax.plot([], [], 'r-', linewidth=6, label='Elo 2')
        base_point, = ax.plot([], [], 'ko', markersize=12, label='Base')
        joint1_point, = ax.plot([], [], 'go', markersize=10, label='Junta 1')
        effector_point, = ax.plot([], [], 'mo', markersize=12, label='Efetuador')
        
        title_text = ax.text(0.5, 1.05, '', transform=ax.transAxes,
                            ha='center', fontsize=12, fontweight='bold')
        
        ax.set_xlim([-0.5, 2.0])
        ax.set_ylim([-1.0, 1.5])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('x - Posição no eixo X (m)', fontsize=12)
        ax.set_ylabel('y - Posição no eixo Y (m)', fontsize=12)
        ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
        ax.set_title('Animação da Trajetória do Braço Robótico 2-DOF', 
                    fontsize=14, fontweight='bold', pad=20)
        
        def init():
            link1_line.set_data([], [])
            link2_line.set_data([], [])
            base_point.set_data([], [])
            joint1_point.set_data([], [])
            effector_point.set_data([], [])
            trace_line.set_data([], [])
            title_text.set_text('')
            return link1_line, link2_line, base_point, joint1_point, effector_point, trace_line, title_text
        
        def update(frame):
            node_id = path_nodes[frame]
            config = graph.node_ids[node_id]
            theta1, theta2 = config
            
            P1, P2 = self.robot.forward_kinematics(theta1, theta2)
            base = np.array([0.0, 0.0])
            
            # Atualiza os elos
            link1_line.set_data([base[0], P1[0]], [base[1], P1[1]])
            link2_line.set_data([P1[0], P2[0]], [P1[1], P2[1]])
            
            # Atualiza as juntas
            base_point.set_data([base[0]], [base[1]])
            joint1_point.set_data([P1[0]], [P1[1]])
            effector_point.set_data([P2[0]], [P2[1]])
            
            # Atualiza o traço
            end_effector_trace[0].append(P2[0])
            end_effector_trace[1].append(P2[1])
            trace_line.set_data(end_effector_trace[0], end_effector_trace[1])
            
            # Atualiza o título com informações mais descritivas
            title_text.set_text(f'Passo {frame+1}/{len(path_nodes)} | ' +
                              f'Junta 1: θ₁ = {theta1:.1f}° | Junta 2: θ₂ = {theta2:.1f}° | ' +
                              f'Efetuador: ({P2[0]:.2f}, {P2[1]:.2f}) m')
            
            return link1_line, link2_line, base_point, joint1_point, effector_point, trace_line, title_text
        
        anim = FuncAnimation(fig, update, init_func=init,
                           frames=len(path_nodes), interval=interval,
                           blit=True, repeat=True)
        
        if save_path:
            print(f"Salvando animação... (isso pode levar alguns minutos)")
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Animação salva em: {save_path}")
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# ANÁLISE E COMPARAÇÃO
# =============================================================================

class ExperimentRunner:
    """Executa experimentos e compara resultados"""
    
    def __init__(self, robot_params: RobotParameters, problem: ProblemInstance):
        self.robot_params = robot_params
        self.problem = problem
        self.results = []
    
    def run_experiment(self, delta_theta: float, cost_function: str,
                      cost_params: dict, algorithm: str, 
                      heuristic: str = None) -> dict:
        """
        Executa um experimento completo
        
        Returns:
            Dicionário com os resultados do experimento
        """
        # Cria instâncias com os parâmetros especificados
        params = RobotParameters()
        params.delta_theta = delta_theta
        params.L1 = self.robot_params.L1
        params.L2 = self.robot_params.L2
        # ... (copia outros parâmetros)
        
        robot = RobotArm(params)
        collision_checker = CollisionChecker(self.problem.obstacles, params.epsilon)
        graph = CSpaceGraph(robot, collision_checker, self.problem)
        
        # Constrói o grafo
        graph.build_graph(cost_function, cost_params)
        
        # Obtém o nó inicial
        start_id = graph.get_start_node()
        if start_id is None:
            print("ERRO: Configuração inicial em colisão!")
            return None
        
        # Executa o algoritmo de busca
        search = SearchAlgorithms(graph)
        
        if algorithm == 'dijkstra':
            path, _ = search.dijkstra(start_id)
        elif algorithm == 'astar':
            path, _ = search.a_star(start_id, heuristic)
        else:
            raise ValueError(f"Algoritmo desconhecido: {algorithm}")
        
        # Coleta resultados
        result = {
            'delta_theta': delta_theta,
            'cost_function': cost_function,
            'cost_params': cost_params,
            'algorithm': algorithm,
            'heuristic': heuristic if algorithm == 'astar' else 'N/A',
            'path_found': path is not None,
            'path_cost': search.path_cost if path else None,
            'nodes_expanded': search.nodes_expanded,
            'search_time': search.search_time,
            'path_length': len(path) if path else 0,
            'total_nodes': graph.free_nodes,
            'graph': graph,
            'path': path
        }
        
        self.results.append(result)
        return result
    
    def print_results_table(self) -> None:
        """Imprime uma tabela formatada com os resultados"""
        print("\n" + "="*120)
        print("RESUMO DOS EXPERIMENTOS")
        print("="*120)
        
        header = f"{'Δθ':<6} {'Custo':<8} {'Algoritmo':<10} {'Heurística':<12} " \
                f"{'Custo':<12} {'Nós Exp.':<10} {'Tempo (s)':<12} {'Path Len':<10}"
        print(header)
        print("-"*120)
        
        for result in self.results:
            if result['path_found']:
                row = f"{result['delta_theta']:<6.1f} " \
                      f"{result['cost_function']:<8} " \
                      f"{result['algorithm']:<10} " \
                      f"{result['heuristic']:<12} " \
                      f"{result['path_cost']:<12.4f} " \
                      f"{result['nodes_expanded']:<10} " \
                      f"{result['search_time']:<12.4f} " \
                      f"{result['path_length']:<10}"
                print(row)
            else:
                print(f"{result['delta_theta']:<6.1f} {result['cost_function']:<8} " \
                      f"{result['algorithm']:<10} {'N/A':<12} {'NO PATH FOUND':<50}")
        
        print("="*120 + "\n")
    
    def plot_comparison(self, save_path: str = None) -> None:
        """Cria gráficos comparativos dos resultados"""
        if not self.results:
            print("Nenhum resultado para plotar!")
            return
        
        # Filtra resultados com caminhos encontrados
        valid_results = [r for r in self.results if r['path_found']]
        
        if not valid_results:
            print("Nenhum caminho encontrado nos experimentos!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepara dados
        labels = []
        costs = []
        nodes_exp = []
        times = []
        path_lens = []
        
        for r in valid_results:
            label = f"{r['algorithm'][:4]}-{r['cost_function']}"
            if r['heuristic'] != 'N/A':
                label += f"-{r['heuristic']}"
            label += f" (Δθ={r['delta_theta']}°)"
            
            labels.append(label)
            costs.append(r['path_cost'])
            nodes_exp.append(r['nodes_expanded'])
            times.append(r['search_time'])
            path_lens.append(r['path_length'])
        
        x = np.arange(len(labels))
        
        # Gráfico 1: Custo do caminho
        axes[0, 0].bar(x, costs, color='steelblue', edgecolor='black', alpha=0.8)
        axes[0, 0].set_xlabel('Configuração do Experimento', fontsize=11)
        axes[0, 0].set_ylabel('Custo Total do Caminho', fontsize=11)
        axes[0, 0].set_title('Custo do Caminho Encontrado\n(Menor é melhor)', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Gráfico 2: Nós expandidos
        axes[0, 1].bar(x, nodes_exp, color='coral', edgecolor='black', alpha=0.8)
        axes[0, 1].set_xlabel('Configuração do Experimento', fontsize=11)
        axes[0, 1].set_ylabel('Número de Nós Expandidos', fontsize=11)
        axes[0, 1].set_title('Eficiência da Busca\n(Menor é mais eficiente)', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Gráfico 3: Tempo de execução
        axes[1, 0].bar(x, times, color='lightgreen', edgecolor='black', alpha=0.8)
        axes[1, 0].set_xlabel('Configuração do Experimento', fontsize=11)
        axes[1, 0].set_ylabel('Tempo de Execução (segundos)', fontsize=11)
        axes[1, 0].set_title('Tempo de Execução do Algoritmo\n(Menor é mais rápido)', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Gráfico 4: Comprimento do caminho
        axes[1, 1].bar(x, path_lens, color='plum', edgecolor='black', alpha=0.8)
        axes[1, 1].set_xlabel('Configuração do Experimento', fontsize=11)
        axes[1, 1].set_ylabel('Número de Passos (Nós no Caminho)', fontsize=11)
        axes[1, 1].set_title('Comprimento do Caminho\n(Número de configurações visitadas)', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.suptitle('Comparação dos Algoritmos de Busca: Dijkstra vs A*\nDiferentes Funções de Custo e Heurísticas', 
                    fontsize=16, fontweight='bold', y=0.998)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráficos comparativos salvos em: {save_path}")
        
        plt.show()


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal para executar os experimentos"""
    
    print("="*80)
    print("PLANEJAMENTO DE MOVIMENTO EM BRAÇO ROBÓTICO 2-DOF")
    print("Algoritmos: Dijkstra e A*")
    print("="*80 + "\n")
    
    # Parâmetros e problema
    robot_params = RobotParameters()
    problem = ProblemInstance()
    
    # Runner de experimentos
    runner = ExperimentRunner(robot_params, problem)
    
    # =========================================================================
    # EXPERIMENTO 1: Dijkstra com Δθ=5° e custo geométrico (c1)
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENTO 1: Dijkstra + Custo Geométrico (c1) + Δθ=5°")
    print("="*80)
    
    result1 = runner.run_experiment(
        delta_theta=5.0,
        cost_function='cost1',
        cost_params={},
        algorithm='dijkstra'
    )
    
    if result1 and result1['path_found']:
        # Visualiza C-space
        algo_info = f"Algoritmo: Dijkstra | Custo: Geométrico (c1) | Δθ=5°"
        result1['graph'].visualize_cspace(
            path_nodes=result1['path'],
            save_path='cspace_dijkstra_cost1_5deg.png',
            algorithm_info=algo_info
        )
        
        # Visualiza workspace
        visualizer = Visualizer(RobotArm(robot_params), problem)
        visualizer.visualize_path_workspace(
            path_nodes=result1['path'],
            graph=result1['graph'],
            num_frames=12,
            save_path='workspace_dijkstra_cost1_5deg.png',
            algorithm_info=algo_info
        )
    
    # =========================================================================
    # EXPERIMENTO 2: A* com heurística h1, Δθ=5° e custo geométrico (c1)
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENTO 2: A* (h1) + Custo Geométrico (c1) + Δθ=5°")
    print("="*80)
    
    result2 = runner.run_experiment(
        delta_theta=5.0,
        cost_function='cost1',
        cost_params={},
        algorithm='astar',
        heuristic='h1'
    )
    
    # =========================================================================
    # EXPERIMENTO 3: A* com heurística h2, Δθ=5° e custo geométrico (c1)
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENTO 3: A* (h2) + Custo Geométrico (c1) + Δθ=5°")
    print("="*80)
    
    result3 = runner.run_experiment(
        delta_theta=5.0,
        cost_function='cost1',
        cost_params={},
        algorithm='astar',
        heuristic='h2'
    )
    
    # =========================================================================
    # EXPERIMENTO 4: Dijkstra com Δθ=5° e custo com clearance (c2)
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENTO 4: Dijkstra + Custo Clearance (c2) + Δθ=5°")
    print("="*80)
    
    result4 = runner.run_experiment(
        delta_theta=5.0,
        cost_function='cost2',
        cost_params={'alpha': 1.0, 'beta': 0.02},
        algorithm='dijkstra'
    )
    
    if result4 and result4['path_found']:
        algo_info = f"Algoritmo: Dijkstra | Custo: Clearance (c2) | Δθ=5°"
        result4['graph'].visualize_cspace(
            path_nodes=result4['path'],
            save_path='cspace_dijkstra_cost2_5deg.png',
            algorithm_info=algo_info
        )
    
    # =========================================================================
    # EXPERIMENTO 5: A* (h1) com Δθ=2° e custo geométrico (c1)
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENTO 5: A* (h1) + Custo Geométrico (c1) + Δθ=2°")
    print("="*80)
    
    result5 = runner.run_experiment(
        delta_theta=2.0,
        cost_function='cost1',
        cost_params={},
        algorithm='astar',
        heuristic='h1'
    )
    
    if result5 and result5['path_found']:
        algo_info = f"Algoritmo: A* (h1) | Custo: Geométrico (c1) | Δθ=2°"
        result5['graph'].visualize_cspace(
            path_nodes=result5['path'],
            save_path='cspace_astar_cost1_2deg.png',
            algorithm_info=algo_info
        )
    
    # =========================================================================
    # EXPERIMENTO 6: Dijkstra com Δθ=5° e custo de energia (c3)
    # =========================================================================
    print("\n" + "="*80)
    print("EXPERIMENTO 6: Dijkstra + Custo Energia (c3) + Δθ=5°")
    print("="*80)
    
    result6 = runner.run_experiment(
        delta_theta=5.0,
        cost_function='cost3',
        cost_params={},
        algorithm='dijkstra'
    )
    
    if result6 and result6['path_found']:
        algo_info = f"Algoritmo: Dijkstra | Custo: Energia (c3) | Δθ=5°"
        result6['graph'].visualize_cspace(
            path_nodes=result6['path'],
            save_path='cspace_dijkstra_cost3_5deg.png',
            algorithm_info=algo_info
        )
    
    # =========================================================================
    # RESUMO E COMPARAÇÕES
    # =========================================================================
    
    # Imprime tabela de resultados
    runner.print_results_table()
    
    # Gráficos comparativos
    runner.plot_comparison(save_path='comparison_plots.png')
    
    # Animação do melhor caminho (A* com h1 e custo c1)
    if result2 and result2['path_found']:
        print("\n" + "="*80)
        print("Criando animação do caminho (A* h1, c1, Δθ=5°)...")
        print("="*80)
        visualizer = Visualizer(RobotArm(robot_params), problem)
        visualizer.animate_path(
            path_nodes=result2['path'],
            graph=result2['graph'],
            save_path='animation_astar_h1.gif',
            interval=100
        )
    
    print("\n" + "="*80)
    print("EXPERIMENTOS CONCLUÍDOS!")
    print("="*80)
    print("\nArquivos gerados:")
    print("  • Mapas do C-space: cspace_*.png")
    print("  • Frames do workspace: workspace_*.png")
    print("  • Gráficos comparativos: comparison_plots.png")
    print("  • Animação: animation_astar_h1.gif")
    print("\n")


if __name__ == "__main__":
    main()