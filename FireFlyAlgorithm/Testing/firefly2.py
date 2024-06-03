import numpy as np

# Firefly Algorithm

# Objective function (replace with your own function)
def objective_function(x):
    return np.sum(x**2)

# Problem dimension
dimension = 10

# Search space bounds
lower_bound = -5
upper_bound = 5

# Population size
population_size = 50

# Maximum number of iterations
max_iterations = 100

# Initialization
population = lower_bound + (upper_bound - lower_bound) * np.random.rand(population_size, dimension)
fitness = np.apply_along_axis(objective_function, 1, population)

# Main loop
for iteration in range(max_iterations):

    # Move fireflies towards brighter ones
    alpha = 0.2  # Attraction coefficient
    beta = 1  # Absorption coefficient
    gamma = 1  # Randomization parameter

    for i in range(population_size):
        for j in range(population_size):
            if fitness[j] < fitness[i]:
                distance = np.linalg.norm(population[i] - population[j])
                attractiveness = np.exp(-gamma * distance**2)
                population[i] += alpha * attractiveness * (population[j] - population[i]) + beta * (np.random.rand(dimension) - 0.5)

        # Limit the updated positions within the search space
        population[i] = np.maximum(lower_bound, population[i])
        population[i] = np.minimum(upper_bound, population[i])

    # Evaluate fitness of the updated population
    fitness = np.apply_along_axis(objective_function, 1, population)

    # Update the best solution and fitness
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]

    # Display the best fitness value at each iteration
    print(f"Iteration {iteration+1}, Best Fitness = {best_fitness}")

# Display the final best solution and fitness
print("-------------------")
print("Optimization Results")
print("-------------------")
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
