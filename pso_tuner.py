import random
from genetic_algorithm import run_ga_experiment

# Search space bounds
BOUNDS = {
    'pop_size': (50.0, 500.0),
    'mut_rate': (0.05, 0.30),
    'crossover_prob': (0.5, 1.0)
}

# Velocity limits (to prevent wild oscillations)
VELOCITY_LIMITS = {
    'pop_size': (-100.0, 100.0),
    'mut_rate': (-0.05, 0.05),
    'crossover_prob': (-0.1, 0.1)
}

# Global Evaluation Cache to memoize GA runs
# Key: (pop_size_snapped, mut_rate_snapped, crossover_prob_snapped)
# Value: average generations (float)
EVALUATION_CACHE = {}
CACHE_HITS = 0
TOTAL_EVALUATIONS = 0

def snap_pop_size(pop_size):
    """
    Snaps population size to the nearest multiple of 50.
    """
    snapped = int(round(pop_size / 50.0)) * 50
    return max(50, min(500, snapped))

def snap_mut_rate(mut_rate):
    """
    Snaps mutation rate to the nearest multiple of 0.05 in [0.05, 0.30].
    """
    snapped = int(round(mut_rate / 0.05)) * 0.05
    return max(0.05, min(0.30, round(snapped, 2)))

def snap_crossover_prob(crossover_prob):
    """
    Snaps crossover probability to the nearest multiple of 0.1 in [0.5, 1.0].
    """
    snapped = int(round(crossover_prob / 0.1)) * 0.1
    return max(0.5, min(1.0, round(snapped, 1)))

class Particle:
    def __init__(self, particle_id):
        self.id = particle_id
        
        # Initialize positions randomly within bounds
        self.pos = {
            'pop_size': random.uniform(*BOUNDS['pop_size']),
            'mut_rate': random.uniform(*BOUNDS['mut_rate']),
            'crossover_prob': random.uniform(*BOUNDS['crossover_prob'])
        }
        
        # Initialize velocities randomly within a fraction of the range
        self.vel = {
            'pop_size': random.uniform(-50.0, 50.0),
            'mut_rate': random.uniform(-0.02, 0.02),
            'crossover_prob': random.uniform(-0.05, 0.05)
        }
        
        # Initial fitness evaluation will happen in the tuning loop
        self.fitness = float('inf')
        
        # Personal best
        self.pbest_pos = self.pos.copy()
        self.pbest_fitness = float('inf')

    def update_position(self):
        for key in self.pos:
            self.pos[key] += self.vel[key]
            
            # Clip position to bounds
            low, high = BOUNDS[key]
            self.pos[key] = max(low, min(high, self.pos[key]))

    def update_velocity(self, gbest_pos, w, c1, c2):
        for key in self.pos:
            r1 = random.random()
            r2 = random.random()
            
            cognitive = c1 * r1 * (self.pbest_pos[key] - self.pos[key])
            social = c2 * r2 * (gbest_pos[key] - self.pos[key])
            
            self.vel[key] = w * self.vel[key] + cognitive + social
            
            # Clip velocity to limits
            low, high = VELOCITY_LIMITS[key]
            self.vel[key] = max(low, min(high, self.vel[key]))

    def to_dict(self):
        return {
            'Particle ID': f"P{self.id}",
            'Population Size': snap_pop_size(self.pos['pop_size']),
            'Mutation Rate': snap_mut_rate(self.pos['mut_rate']),
            'Crossover Prob': snap_crossover_prob(self.pos['crossover_prob']),
            'Fitness (Avg Gen)': round(self.fitness, 2) if self.fitness != float('inf') else "N/A",
            'Vel Pop': round(self.vel['pop_size'], 2),
            'Vel Mut': round(self.vel['mut_rate'], 4),
            'Vel Cross': round(self.vel['crossover_prob'], 3),
            'Best Pop': snap_pop_size(self.pbest_pos['pop_size']),
            'Best Mut': snap_mut_rate(self.pbest_pos['mut_rate']),
            'Best Cross': snap_crossover_prob(self.pbest_pos['crossover_prob']),
            'Best Fitness': round(self.pbest_fitness, 2) if self.pbest_fitness != float('inf') else "N/A"
        }

def evaluate_fitness_with_cache(pos, target_num, max_gens, num_runs):
    """
    Evaluates fitness using cache lookup. Snaps all three search parameters.
    """
    global CACHE_HITS, TOTAL_EVALUATIONS
    
    # Snap search coordinates for cache lookup and GA run
    pop_size_snapped = snap_pop_size(pos['pop_size'])
    mut_rate_snapped = snap_mut_rate(pos['mut_rate'])
    crossover_prob_snapped = snap_crossover_prob(pos['crossover_prob'])
    
    cache_key = (pop_size_snapped, mut_rate_snapped, crossover_prob_snapped)
    TOTAL_EVALUATIONS += 1
    
    if cache_key in EVALUATION_CACHE:
        CACHE_HITS += 1
        return EVALUATION_CACHE[cache_key]
    
    # Calculate fitness by running the GA
    total_gens = 0
    for _ in range(num_runs):
        gens, _ = run_ga_experiment(
            pop_size=pop_size_snapped,
            mut_rate=mut_rate_snapped,
            target_num=target_num,
            crossover_probability=crossover_prob_snapped,
            max_generations=max_gens
        )
        total_gens += gens
        
    avg_gens = total_gens / num_runs
    EVALUATION_CACHE[cache_key] = avg_gens
    return avg_gens

def run_pso_tuner_generator(
    target_num,
    num_particles=10,
    max_iterations=12,
    num_runs_per_eval=3,
    max_gens_limit=500,
    w=0.7,
    c1=2.0,
    c2=1.0
):
    """
    A generator that runs the PSO hyperparameter tuning algorithm.
    Yields intermediate state on each iteration for real-time visualization.
    """
    global CACHE_HITS, TOTAL_EVALUATIONS
    CACHE_HITS = 0
    TOTAL_EVALUATIONS = 0
    
    particles = [Particle(i) for i in range(num_particles)]
    gbest_pos = {}
    gbest_fitness = float('inf')
    
    # ----------------------------------------------------
    # ITERATION 0: Evaluate initial random positions
    # ----------------------------------------------------
    for p in particles:
        p.fitness = evaluate_fitness_with_cache(p.pos, target_num, max_gens_limit, num_runs_per_eval)
        p.pbest_fitness = p.fitness
        p.pbest_pos = p.pos.copy()
        
        if p.fitness < gbest_fitness:
            gbest_fitness = p.fitness
            gbest_pos = p.pos.copy()
            
    yield {
        'iteration': 0,
        'particles': [p.to_dict() for p in particles],
        'raw_particles': particles,
        'gbest_pos': gbest_pos.copy(),
        'gbest_fitness': gbest_fitness,
        'cache_hits': CACHE_HITS,
        'total_evaluations': TOTAL_EVALUATIONS
    }

    # ----------------------------------------------------
    # ITERATIONS 1 to max_iterations
    # ----------------------------------------------------
    for it in range(1, max_iterations + 1):
        for p in particles:
            # Update velocities and positions
            p.update_velocity(gbest_pos, w, c1, c2)
            p.update_position()
            
            # Evaluate new positions
            p.fitness = evaluate_fitness_with_cache(p.pos, target_num, max_gens_limit, num_runs_per_eval)
            
            # Update personal best
            if p.fitness < p.pbest_fitness:
                p.pbest_fitness = p.fitness
                p.pbest_pos = p.pos.copy()
                
                # Update global best
                if p.fitness < gbest_fitness:
                    gbest_fitness = p.fitness
                    gbest_pos = p.pos.copy()
                    
        yield {
            'iteration': it,
            'particles': [p.to_dict() for p in particles],
            'raw_particles': particles,
            'gbest_pos': gbest_pos.copy(),
            'gbest_fitness': gbest_fitness,
            'cache_hits': CACHE_HITS,
            'total_evaluations': TOTAL_EVALUATIONS
        }
