import random

def create_chromosome(length):
    """
    Creates a random individual (chromosome) consisting of integers 
    from 1 to 26 representing the alphabet (A=1, B=2, ..., Z=26).
    """
    return [random.randint(1, 26) for _ in range(length)]

def calculate_fitness(chromosome, target_num):
    """
    Calculates how close a chromosome is to the target.
    It computes the absolute difference between corresponding letters.
    The formula (max_possible - difference) ensures that a higher score means a better match.
    """
    difference = sum([abs(g - t) for g, t in zip(chromosome, target_num)])
    return (len(target_num) * 26) - difference

def roulette_wheel_selection(population, fitnesses, total_fitness):
    """
    Selects an individual from the population using the Roulette Wheel method.
    Individuals with higher fitness have a proportionally higher chance of being selected.
    """
    if total_fitness == 0:
        return random.choice(population)
    
    # Pick a random point on the 'wheel'
    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitnesses):
        current += fitness
        if current > pick:
            return individual
    return population[-1]

def crossover(parent1, parent2, target_num, crossover_probability):
    """
    Combines two parents to create two children (Single-Point Crossover).
    A random cutoff point is chosen, and the genetic material is swapped.
    """
    if len(target_num) <= 1 or random.random() >= crossover_probability:
        return parent1[:], parent2[:]
        
    crossover_point = random.randint(1, len(target_num) - 1)
    return (parent1[:crossover_point] + parent2[crossover_point:],
            parent2[:crossover_point] + parent1[crossover_point:])

def mutate(chromosome, mutation_rate):
    """
    Randomly changes some genes (letters) in the chromosome based on the mutation rate.
    This helps maintain genetic diversity and prevents getting stuck in local optima.
    """
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.randint(1, 26)
    return chromosome

def run_ga_experiment(pop_size, mut_rate, target_num, crossover_probability, max_generations=500):
    """
    Runs the genetic algorithm for a specific set of parameters.
    Uses a (Mu + Lambda) selection strategy where parents and children compete.
    
    Parameters can be floats (passed by PSO). They are coerced and validated.
    """
    # Coerce parameters from PSO to valid types/ranges
    pop_size = max(4, int(round(pop_size)))
    mut_rate = max(0.0, min(1.0, float(mut_rate)))
    crossover_probability = max(0.0, min(1.0, float(crossover_probability)))

    # Initialize the starting population with random chromosomes and their fitness
    population = []
    for _ in range(pop_size):
        chrom = create_chromosome(len(target_num))
        fit = calculate_fitness(chrom, target_num)
        population.append((chrom, fit))
        
    # Keep population sorted by fitness descending
    population.sort(key=lambda x: x[1], reverse=True)
    
    generation = 0
    history_log = []
    MAX_POSSIBLE_FITNESS = len(target_num) * 26

    while generation < max_generations:
        generation += 1
        
        # Best individual is at index 0 because population is kept sorted
        best_chromosome, best_fitness = population[0]

        # Convert the best numeric chromosome back to a string and log it
        best_string = "".join(chr(g + 64) for g in best_chromosome)
        best_numbers = " ".join(f"{g:2}" for g in best_chromosome)
        
        history_log.append(f"Gen {generation:02d}: [{best_numbers}] --- {best_string} (Fit: {best_fitness})")

        # Termination condition: Stop if the perfect match is found
        if best_fitness == MAX_POSSIBLE_FITNESS:
            return generation, history_log

        # Extract fitnesses for selection weights
        fitnesses = [fit for _, fit in population]

        # Generate the next generation of children
        children = []
        # We need pop_size children. Each crossover produces 2 children.
        # So we need pop_size // 2 pairs.
        num_pairs = (pop_size + 1) // 2
        
        # Select parents in one fast O(pop_size) operation
        parent_pairs = random.choices(population, weights=fitnesses, k=2 * num_pairs)
        
        for i in range(0, 2 * num_pairs, 2):
            p1, _ = parent_pairs[i]
            p2, _ = parent_pairs[i+1]
            c1, c2 = crossover(p1, p2, target_num, crossover_probability)
            c1 = mutate(c1, mut_rate)
            c2 = mutate(c2, mut_rate)
            children.append((c1, calculate_fitness(c1, target_num)))
            children.append((c2, calculate_fitness(c2, target_num)))

        # Ensure we don't exceed the intended population size
        children = children[:pop_size]
        
        # Elitism / Survivor Selection: Combine parents and children
        combined_population = population + children
        # Sort by precomputed fitness value (index 1 of the tuple) - extremely fast!
        combined_population.sort(key=lambda x: x[1], reverse=True)
        population = combined_population[:pop_size]

    # If target is not matched, calculate a penalty score.
    # Penalty: max_generations + (remaining distance to target * 10)
    best_fitness = population[0][1]
    unfitness = MAX_POSSIBLE_FITNESS - best_fitness
    penalty_score = max_generations + (unfitness * 10)
    
    return penalty_score, history_log
