import random
import numpy as np
import math

# === Parameters ===
POP_SIZE = 50  # Number of individuals in the population
GENES = 8      # Number of genes per individual
GENERATIONS = 30  # Number of generations to run
ELITE_COUNT = 3   # Number of elite individuals to keep each generation
CROSSOVER_RATE = 0.9  # Probability of crossover
MUTATION_RATE = 0.1   # Probability of mutation per gene
TOURNAMENT_SIZE = 3   # Number of individuals in tournament selection
INITIAL_MUTATION_STD = 0.5  # Initial stddev for Gaussian mutation
MUTATION_DECAY = 0.95       # Decay factor for mutation stddev
INITIAL_SBX_ETA = 2         # Initial eta for SBX (not used in uniform crossover)
SBX_ETA_GROWTH = 1.05       # Growth factor for SBX eta
POLY_MUTATION_ETA = 20      # Eta for polynomial mutation

# === Fixed input individuals for blend ===
INPUT_A = [0.9, 0.9, 0.0, 0.8, 0.2, 0.9, 0.7, 0.7]
INPUT_B = [0.8, 0.2, 1.0, 0.3, 0.9, 0.4, 0.6, 0.3]

# === Fitness Function ===
def fitness(candidate):
    """
    Calculates the fitness of a candidate individual.
    The fitness is based on the emergence and contribution of each gene compared to two fixed input individuals.
    """
    emergence = [c - max(a, b) for c, a, b in zip(candidate, INPUT_A, INPUT_B)]
    emergence = [max(0, e) for e in emergence]  # Clamp negative emergence to 0
    contributions = [min(a, b) * e for a, b, e in zip(INPUT_A, INPUT_B, emergence)]
    total = sum(contributions)
    return min(total / GENES, 1.0)

# === Initialization ===
def initialize_population():
    """
    Initializes the population with random individuals.
    Each gene is a float in [0, 1].
    """
    return [np.random.uniform(0, 1, GENES).tolist() for _ in range(POP_SIZE)]

# === Selection: Tournament Selection ===
def tournament_selection(population, fitnesses, tournament_size=TOURNAMENT_SIZE):
    """
    Selects an individual using tournament selection.
    Randomly picks 'tournament_size' individuals and returns the one with the highest fitness.
    """
    selected_indices = random.sample(range(len(population)), tournament_size)
    best_idx = max(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_idx]

# === Selection: Vectorized Roulette-Wheel Selection ===
def roulette_wheel_selection(population, fitnesses):
    """
    Selects an individual using roulette-wheel selection (fitness-proportionate selection).
    This implementation is vectorized and does not use a while loop.
    """
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        # If all fitnesses are zero, select randomly
        return random.choice(population)
    probs = np.array(fitnesses) / total_fitness
    idx = np.random.choice(len(population), p=probs)
    return population[idx]

# === Crossover: Uniform Crossover ===
def uniform_crossover(parent1, parent2):
    """
    Performs uniform crossover between two parents.
    Each gene is independently chosen from one of the parents.
    """
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]
    child1, child2 = [], []
    for g1, g2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(g1)
            child2.append(g2)
        else:
            child1.append(g2)
            child2.append(g1)
    return child1, child2

# === Mutation: Polynomial Mutation ===
def polynomial_mutation(individual, eta=POLY_MUTATION_ETA):
    """
    Applies polynomial mutation to an individual.
    Each gene has a chance to be mutated according to the polynomial mutation formula.
    """
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            x = individual[i]
            r = random.random()
            if r < 0.5:
                delta = (2 * r) ** (1.0 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - r)) ** (1.0 / (eta + 1))
            x += delta
            # Ensure gene stays within [0, 1]
            individual[i] = min(max(x, 0.0), 1.0)
    return individual

# === Main GA Loop ===
def genetic_algorithm(selection_fn, crossover_fn, mutation_fn):
    """
    Runs the genetic algorithm using the provided selection, crossover, and mutation functions.
    """
    population = initialize_population()
    mutation_std = INITIAL_MUTATION_STD
    sbx_eta = INITIAL_SBX_ETA  # Not used in uniform crossover, but kept for compatibility

    for gen in range(GENERATIONS):
        fitnesses = [fitness(ind) for ind in population]
        # Elitism: keep the best individuals
        elites = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:ELITE_COUNT]
        new_population = [ind for ind, _ in elites]

        # Generate new individuals
        while len(new_population) < POP_SIZE:
            parent1 = selection_fn(population, fitnesses)
            parent2 = selection_fn(population, fitnesses)
            child1, child2 = crossover_fn(parent1, parent2)
            child1 = mutation_fn(child1)
            child2 = mutation_fn(child2)
            new_population.extend([child1, child2])

        population = new_population[:POP_SIZE]  # Ensure population size remains constant
        best_fitness = max(fitnesses)
        print(f"Generation {gen+1}: Best Fitness = {best_fitness:.4f}")

        mutation_std *= MUTATION_DECAY  # Decay mutation over time (if using Gaussian mutation)
        sbx_eta *= SBX_ETA_GROWTH       # Increase SBX eta over time (if using SBX)

    # Final result
    final_fitnesses = [fitness(ind) for ind in population]
    best = max(zip(population, final_fitnesses), key=lambda x: x[1])
    print("\nBest Individual:", best[0])
    print("Best Fitness:", best[1])

if __name__ == "__main__":
    print("\n--- Running GA with Tournament Selection, Uniform Crossover, Polynomial Mutation ---\n")
    genetic_algorithm(
        selection_fn=tournament_selection,
        crossover_fn=uniform_crossover,
        mutation_fn=polynomial_mutation
    )

    print("\n--- Running GA with Vectorized Roulette-Wheel Selection, Uniform Crossover, Polynomial Mutation ---\n")
    genetic_algorithm(
        selection_fn=roulette_wheel_selection,
        crossover_fn=uniform_crossover,
        mutation_fn=polynomial_mutation
    ) 