import numpy as np
import tensorflow as tf
import random
import pandas as pd
import time
from collections import deque


np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class GDPSOScheduler:
    def __init__(self, tasks, fog_nodes, population_size=100, 
                 ga_iterations=50, ddpg_episodes=30, pso_iterations=20,
                 crossover_rate=0.7, mutation_rate=0.1):
        """
        Initialize the GDPSO scheduler with tasks and fog nodes.
        
        Args:
            tasks: List of tasks with execution_time, deadline, priority, etc.
            fog_nodes: List of fog nodes with processing capacity, reliability, etc.
            population_size: Size of genetic population
            ga_iterations: Number of GA iterations
            ddpg_episodes: Number of DDPG episodes
            pso_iterations: Number of PSO iterations
            crossover_rate: GA crossover probability
            mutation_rate: GA mutation probability
        """
        self.tasks = tasks
        self.fog_nodes = fog_nodes
        self.num_tasks = len(tasks)
        self.num_nodes = len(fog_nodes)
        self.population_size = population_size
        self.ga_iterations = ga_iterations
        self.ddpg_episodes = ddpg_episodes
        self.pso_iterations = pso_iterations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        
        self.initialize_ddpg()
        
 
        self.best_solution = None
        self.best_fitness = float('inf')
        self.performance_history = []
        
    def initialize_ddpg(self):
        """Initialize DDPG actor and critic networks"""

        self.actor = self.build_actor_network()
        self.target_actor = self.build_actor_network()
        
        
        self.critic = self.build_critic_network()
        self.target_critic = self.build_critic_network()
     
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
       
        self.replay_buffer = deque(maxlen=10000)
        
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
    def build_actor_network(self):
        """Build the actor network for DDPG"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.num_tasks + self.num_nodes,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_nodes, activation='softmax')
        ])
        return model
        
    def build_critic_network(self):
        """Build the critic network for DDPG"""
        state_input = tf.keras.layers.Input(shape=(self.num_tasks + self.num_nodes,))
        action_input = tf.keras.layers.Input(shape=(self.num_nodes,))
        
        state_h1 = tf.keras.layers.Dense(256, activation='relu')(state_input)
        state_h2 = tf.keras.layers.Dense(128, activation='relu')(state_h1)
        
        action_h1 = tf.keras.layers.Dense(128, activation='relu')(action_input)
        
        merged = tf.keras.layers.Concatenate()([state_h2, action_h1])
        merged_h1 = tf.keras.layers.Dense(64, activation='relu')(merged)
        output = tf.keras.layers.Dense(1)(merged_h1)
        
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=output)
        return model
    
    def run_gdpso(self):
        """Execute the full GDPSO algorithm pipeline"""
        start_time = time.time()
        
      
        population = self.initialize_population()
        for i in range(self.ga_iterations):
            population = self.ga_phase(population)
            best_chromosome = min(population, key=lambda x: self.calculate_fitness(x))
            self.update_best_solution(best_chromosome)
        
       
        top_chromosomes = sorted(population, key=lambda x: self.calculate_fitness(x))[:20]
        
     
        policy_solutions = self.ddpg_phase(top_chromosomes)
        
       
        best_policies = sorted(policy_solutions, key=lambda x: self.calculate_fitness(x))[:20]
        

        final_solution = self.pso_phase(best_policies)
        self.update_best_solution(final_solution)
        
        execution_time = time.time() - start_time
        print(f"GDPSO completed in {execution_time:.2f} seconds")
        
        return self.best_solution, self.calculate_performance_metrics(self.best_solution)
        
    def initialize_population(self):
        """Initialize genetic algorithm population"""
        population = []
        for _ in range(self.population_size):
          
            chromosome = np.random.randint(0, self.num_nodes, size=self.num_tasks)
            population.append(chromosome)
        return population
    
    def ga_phase(self, population):
        """Execute one generation of the genetic algorithm"""
        
        fitness_values = [self.calculate_fitness(chromosome) for chromosome in population]
        
  
        new_population = []
        
        elite_indices = np.argsort(fitness_values)[:2]
        for idx in elite_indices:
            new_population.append(population[idx])
        
        while len(new_population) < self.population_size:
      
            parent1 = self.tournament_selection(population, fitness_values)
            parent2 = self.tournament_selection(population, fitness_values)
            
        
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        return new_population
    
    def tournament_selection(self, population, fitness_values, tournament_size=3):
        """Tournament selection for GA"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1, parent2):
        """Two-point crossover"""
        if len(parent1) < 2:
            return parent1.copy(), parent2.copy()
        
        point1, point2 = sorted(random.sample(range(len(parent1)), 2))
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        child1[point1:point2] = parent2[point1:point2]
        child2[point1:point2] = parent1[point1:point2]
        
        return child1, child2
    
    def mutation(self, chromosome):
        """Random mutation"""
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.randint(0, self.num_nodes - 1)
        return chromosome
    
    def ddpg_phase(self, initial_solutions):
        """DDPG phase for policy optimization"""
        # Convert chromosomes to initial states
        states = [self.solution_to_state(solution) for solution in initial_solutions]
        refined_solutions = []
        
        for episode in range(self.ddpg_episodes):
            for state_idx, initial_state in enumerate(states):
                state = initial_state.copy()
                solution = initial_solutions[state_idx].copy()
                
                for t in range(self.num_tasks):
                    
                    action = self.get_action(state)
                    
                 
                    solution[t] = np.argmax(action)
                    next_state = self.get_next_state(state, action, t)
                    
                   
                    reward = -self.calculate_fitness(solution) 
                    
                   
                    self.replay_buffer.append((state, action, reward, next_state, t == self.num_tasks - 1))
               
                    state = next_state
                    
              
                    self.train_ddpg()
                
                refined_solutions.append(solution)
                
       
            self.update_target_networks()
        
        return refined_solutions
    
    def solution_to_state(self, solution):
        """Convert a task allocation solution to a state vector"""
    
        task_features = np.array([
            [
                self.tasks[i]['execution_time'] / 100.0,
                self.tasks[i]['deadline'] / 1000.0,
                self.tasks[i]['priority'] / 5.0
            ] for i in range(self.num_tasks)
        ]).flatten()
        
    
        node_features = np.array([
            self.fog_nodes[i]['reliability'] for i in range(self.num_nodes)
        ])
        

        return np.concatenate([task_features, node_features])
    
    def get_action(self, state):
        """Get action from actor network with exploration noise"""
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor(state_tensor)[0].numpy()
        
        
        noise = np.random.normal(0, 0.1, size=self.num_nodes)
        action = action + noise
        
       
        action = np.clip(action, 0, 1)
        action = action / np.sum(action)
        
        return action
    
    def get_next_state(self, state, action, task_idx):
        """Calculate next state after applying action"""
        next_state = state.copy()
        
      
        node_idx = np.argmax(action)
        next_state[self.num_tasks*3 + node_idx] += self.tasks[task_idx]['execution_time'] / 1000.0
        
        return next_state
    
    def train_ddpg(self):
        """Train DDPG networks using sampled mini-batch"""
        if len(self.replay_buffer) < 64:
            return
        
    
        mini_batch = random.sample(self.replay_buffer, 64)
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        with tf.GradientTape() as tape:
      
            q_values = self.critic([states, actions], training=True)
            
         
            next_actions = self.target_actor(next_states)
            
        
            next_q_values = self.target_critic([next_states, next_actions])
            
          
            target_q = rewards + (1 - dones) * 0.99 * next_q_values
            
          
            critic_loss = tf.reduce_mean(tf.square(target_q - q_values))
        
 
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        

        with tf.GradientTape() as tape:
            actions = self.actor(states)
            q_values = self.critic([states, actions])
            actor_loss = -tf.reduce_mean(q_values)
        
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
    
    def update_target_networks(self, tau=0.005):
        """Update target networks with soft update"""
        for target_var, var in zip(self.target_actor.variables, self.actor.variables):
            target_var.assign(tau * var + (1 - tau) * target_var)
        
        for target_var, var in zip(self.target_critic.variables, self.critic.variables):
            target_var.assign(tau * var + (1 - tau) * target_var)
    
    def pso_phase(self, initial_solutions):
        """PSO phase for final refinement"""
      
        particles = np.array(initial_solutions)
        velocities = np.zeros((len(particles), self.num_tasks))
        
    
        w = 0.9  
        c1 = 2.05 
        c2 = 2.05 
        
        
        p_best = particles.copy()
        p_best_fitness = np.array([self.calculate_fitness(p) for p in p_best])
        g_best_idx = np.argmin(p_best_fitness)
        g_best = p_best[g_best_idx].copy()
        
    
        for i in range(self.pso_iterations):
      
            w = 0.9 - 0.5 * (i / self.pso_iterations)
            
            for j in range(len(particles)):
        
                r1, r2 = random.random(), random.random()
                cognitive = c1 * r1 * (p_best[j] - particles[j])
                social = c2 * r2 * (g_best - particles[j])
                velocities[j] = w * velocities[j] + cognitive + social
                
               
                v_discrete = np.round(velocities[j]).astype(int)
                
                 
                new_position = particles[j] + v_discrete
                
             
                new_position = np.clip(new_position, 0, self.num_nodes - 1)
                
             
                particles[j] = new_position
                
              
                fitness = self.calculate_fitness(particles[j])
                if fitness < p_best_fitness[j]:
                    p_best[j] = particles[j].copy()
                    p_best_fitness[j] = fitness
                    
            
                    if fitness < self.calculate_fitness(g_best):
                        g_best = particles[j].copy()
            
          
            self.check_resource_utilization(g_best)
        
        return g_best
    
    def check_resource_utilization(self, solution):
        """Check if resource utilization exceeds threshold and provision new nodes if needed"""
        node_loads = np.zeros(self.num_nodes)
        for i, node_idx in enumerate(solution):
            node_loads[node_idx] += self.tasks[i]['execution_time']
        
        
        avg_load = np.sum(node_loads) / self.num_nodes
        max_load = np.max(node_loads)
        

        if max_load > 1.8 * avg_load and len(self.fog_nodes) < self.num_nodes + 5:
      
            new_node = {
                'id': len(self.fog_nodes),
                'capacity': np.mean([n['capacity'] for n in self.fog_nodes]),
                'reliability': 0.9  
            }
            self.fog_nodes.append(new_node)
            self.num_nodes += 1
    
    def calculate_fitness(self, solution):
        """
        Calculate fitness (lower is better)
        Combines makespan, trust, and scalability
        """
        makespan = self.calculate_makespan(solution)
        trust = self.calculate_trust(solution)
        scalability = self.calculate_scalability(solution)
        
        
        fitness = 0.4 * makespan + 0.3 * (1 - trust) + 0.3 * scalability
        return fitness
    
    def calculate_makespan(self, solution):
        """Calculate makespan (completion time) for a solution"""
        node_completion_times = np.zeros(self.num_nodes)
        
        for i, node_idx in enumerate(solution):
            # Add task execution time to node
            node_completion_times[node_idx] += self.tasks[i]['execution_time']
        
        return np.max(node_completion_times)
    
    def calculate_trust(self, solution):
        """Calculate trust coefficient for a solution"""
        completed_tasks = 0
        total_tasks = len(self.tasks)
        
        for i, node_idx in enumerate(solution):
          
            if random.random() < self.fog_nodes[node_idx]['reliability']:
                completed_tasks += 1
        
      
        node_reliability = np.mean([node['reliability'] for node in self.fog_nodes])
        
        return (completed_tasks / total_tasks) * node_reliability
    
    def calculate_scalability(self, solution):
        """Calculate scalability factor for a solution"""
       
        base_makespan = self.calculate_makespan(solution)
        
        
        doubled_makespan = base_makespan * 2.2 
        
        
        return doubled_makespan / (base_makespan * 2.0)
    
    def update_best_solution(self, solution):
        """Update the best solution if current one is better"""
        fitness = self.calculate_fitness(solution)
        if fitness < self.best_fitness:
            self.best_solution = solution.copy()
            self.best_fitness = fitness
            print(f"New best solution found with fitness: {fitness:.4f}")
    
    def calculate_performance_metrics(self, solution):
        """Calculate all performance metrics for a solution"""
        makespan = self.calculate_makespan(solution)
        trust = self.calculate_trust(solution)
        scalability = self.calculate_scalability(solution)
        
        return {
            'makespan': makespan,
            'trust': trust,
            'scalability': scalability
        }
