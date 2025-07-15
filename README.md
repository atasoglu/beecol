# :bee: beecol

A simple Artificial Bee Colony (ABC) algorithm library for Python.

## Features

- Pure Python implementation of the Artificial Bee Colony (ABC) algorithm
- Simple API for continuous optimization problems
- Easily extensible and customizable
- Only dependency: `numpy`

## Installation

```bash
pip install beecol
```

Or clone this repository and use locally:

```bash
git clone https://github.com/atasoglu/beecol.git
cd beecol
pip install .
```

## Usage

```python
import numpy as np
from beecol import ArtificialBeeColony

# Define your fitness function (to maximize)
def fit_func(x):
    # Example: Sphere function (min at 0, but ABC maximizes, so use negative)
    return -np.sum(x**2)

# Set up the optimizer
abc = ArtificialBeeColony(
    fit_func=fit_func,
    dim=5,                # Number of parameters
    bounds=(-5, 5),       # Search space bounds
    n_bees=20,            # Number of bees (food sources)
)

# Run optimization
for i in range(100):
    best_solution, best_fitness = abc.step()
    print(f"Iteration {i}: Best fitness = {best_fitness}")

print("Best solution found:", best_solution)
```

### Example: Knapsack Problem

A full example is provided in [`examples/knapsack.py`](examples/knapsack.py), including plotting and a custom fitness function for the 0/1 knapsack problem.

To run:

```bash
cd examples
python knapsack.py
```

## Citation

- Karaboga, D., & Basturk, B. (2007). *A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm*. Journal of Global Optimization, 39(3), 459–471.
- Karaboga, D., & Basturk, B. (2007). *Artificial Bee Colony (ABC) optimization algorithm for solving constrained optimization problems*. In *Foundations of Fuzzy Logic and Soft Computing*, LNCS 4529, 789–798.


## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to help improve pygena.

## License

MIT