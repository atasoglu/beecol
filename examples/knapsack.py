import numpy as np
from beecol import ArtificialBeeColony

# Set up random number generator for reproducibility
rgen = np.random.default_rng(42)
dim = 20  # Number of items in the knapsack problem
weights = rgen.uniform(1, 20, dim)  # Random weights for each item
values = rgen.uniform(1, 10, dim)  # Random values for each item
capacity = 40  # Maximum weight capacity of the knapsack
bounds = (0, 1)  # Each item can be either included (1) or not (0)
n_bees = 20  # Number of bees (food sources) in the ABC algorithm
n_iter = 100  # Number of optimization iterations


def plot_result(
    fitness_result: np.ndarray,
    weights_result: np.ndarray,
    capacity: float,
) -> None:
    """
    Plot the optimization results for the knapsack problem.

    Args:
        fitness_result (np.ndarray): Array of best fitness values per iteration.
        weights_result (np.ndarray): Array of best solution weights per iteration.
        capacity (float): The knapsack's maximum weight capacity (for reference line).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib with `pip install matplotlib` to plot the result.")
        return None

    fig, (f, w) = plt.subplots(nrows=2, ncols=1, figsize=(6, 4))
    fig.suptitle("Knapsack problem solving")
    # Plotting fitness result
    f.grid(True)
    f.plot(fitness_result, "-b", label="Best Fitness")
    f.set_xlabel("Iteration")
    f.set_ylabel("Fitness")
    f.legend()
    # Plotting weights result
    w.grid(True)
    w.axhline(y=capacity, color="r", linestyle="--", label="Capacity")
    w.plot(weights_result, "-g", label="Best Weight")
    w.set_xlabel("Iteration")
    w.set_ylabel("Weight")
    w.legend()
    plt.tight_layout()
    plt.show()


def fit_func(solution: np.ndarray) -> float:
    """
    Fitness function for the knapsack problem.

    Args:
        solution (np.ndarray): A vector of real values (0-1) representing item selection probabilities.

    Returns:
        float: The total value of the selected items if within capacity, otherwise a negative penalty.
    """
    # Convert continuous solution to binary (0 or 1) for item selection
    solution_bool = np.round(solution)
    total_weight = np.sum(solution_bool * weights)
    total_value = np.sum(solution_bool * values)

    # If weight exceeds capacity, penalize the solution
    if total_weight > capacity:
        # Return a negative fitness to indicate invalid solution
        # The penalty is proportional to how much the capacity is exceeded
        penalty = total_weight - capacity
        return -penalty

    # Return the total value as fitness for valid solutions
    return total_value


def main():
    """
    Main function to solve the knapsack problem using the Artificial Bee Colony algorithm.
    Initializes the optimizer, runs the optimization loop, and plots the results.
    """
    # Initialize the ABC optimizer with the knapsack fitness function
    abc = ArtificialBeeColony(
        fit_func=fit_func,
        dim=dim,
        bounds=bounds,
        n_bees=n_bees,
    )

    fitness_result = []  # Stores best fitness at each iteration
    weights_result = []  # Stores best solution weight at each iteration

    # Optimization loop
    for i in range(n_iter):
        # Perform one optimization step (iteration)
        best_solution, best_fitness = abc.step()
        # Convert solution to binary (0 or 1) for item selection
        best_solution = np.round(best_solution).astype(int)
        # Calculate total weight of the selected items
        best_weight = np.sum(weights[best_solution])
        # Print progress for each iteration
        print(f"Iteration: {i}")
        print(f"Best solution: {best_solution}")
        print(f"Best fitness: {best_fitness}")
        print(f"Best weight: {best_weight}")
        # Store results for plotting
        fitness_result.append(best_fitness)
        weights_result.append(np.sum(best_solution * weights))
    # Plot the optimization results
    plot_result(fitness_result, weights_result, capacity)


if __name__ == "__main__":
    main()
