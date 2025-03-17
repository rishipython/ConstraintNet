# ConstraintNet
Implementing Constraint Modules for Neural Network Outputs

This repository contains a PyTorch-based module for enforcing feasibility constraints on the outputs of a transformer (or other neural network) model. The project demonstrates how to post-process model outputs so that they obey hard constraints during training. The constraints include:

- **Range Constraints**: Clamp each product's output to lie within a specified minimum and maximum range.
- **Mutual Exclusivity Constraints**: Ensure that certain products cannot be nonzero simultaneously.
- **Ranking Constraints**: Enforce that one product’s output is always less than or equal to another’s, based on a given ranking.

The module is implemented in a differentiable manner, so it can be integrated seamlessly with model training. The repository also includes a comprehensive test suite (with both standard unit tests and adversarial tests) that verifies the correctness and robustness of each submodule.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Integrating the Constraint Module](#integrating-the-constraint-module)
  - [Running Tests](#running-tests)
  - [Adversarial Tests](#adversarial-tests)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Differentiable Constraint Enforcement**: Implements hard constraints on the model outputs while preserving the differentiability required for gradient-based training.
- **Multiple Constraints**:
  - *Range Constraints*: Clamps outputs to valid ranges.
  - *Mutual Exclusivity*: Prevents incompatible outputs from being active simultaneously.
  - *Ranking Constraints*: Ensures the ordering of outputs as specified.
- **Robust Testing**:
  - Standard unit tests using `pytest`.
  - Adversarial tests that use gradient descent to attempt to “break” the constraints.
- **Flexible Configuration**: Supports various constraint configurations with dynamic test setups.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/rishipython/ConstraintNet.git
   cd ConstraintNet
   ```

2. **Create and Activate a Conda Environment (Recommended):**

   ```bash
   conda create -n constraintnet python=3.10
   conda activate constraintnet
   ```

3. **Install Requirements:**

   The project depends on PyTorch and pytest. You can install them using pip (or conda for PyTorch):

   ```bash
   pip install torch pytest
   ```

   Or, if you prefer conda for PyTorch:

   ```bash
   conda install pytorch torchvision torchaudio -c pytorch
   pip install pytest
   ```

## Usage

### Integrating the Constraint Module

The main module is implemented in `priority_approach/constraints.py` and provides the following classes:

- `RangeConstraint`
- `MutualExclusivityConstraint`
- `RankingConstraint`
- `ConstraintModule`: A wrapper that applies the three constraints sequentially.

You can use the `ConstraintModule` as a post-processing layer for your model's outputs. The expected input shape is `(batch_size, sequence_length, 2 * number_of_products)`, where the first half represents predicted schedules and the second half represents priority values (used for resolving mutual exclusivity).

Example:

```python
import torch
from priority_approach.constraints import ConstraintModule

# Define constraint parameters
product_ranges = torch.tensor([[0, 1], [0.5, 2], [1, 3], [1.5, 4]])
exclusivities = torch.tensor([[0, 1], [2, 3]])
product_rankings = torch.tensor([[0, 2], [1, 3]])

# Create the ConstraintModule
constraint_module = ConstraintModule(
    number_of_products=4,
    product_ranges=product_ranges,
    exclusivities=exclusivities,
    product_rankings=product_rankings
)

# Example model output (dummy data)
input_tensor = torch.rand(8, 10, 8)  # (batch, sequence_length, 2*number_of_products)

# Enforce constraints
constrained_output = constraint_module(input_tensor)
```

### Running Tests

The repository includes a comprehensive test suite using `pytest`.

To run the tests, execute:

```bash
cd priority_approach
pytest -v
```

If you want to see print statements and logging output, use:

```bash
pytest -s --log-cli-level=INFO
```

### Adversarial Tests

In addition to standard unit tests, there is a separate test file (`test_adversarial.py`) that uses gradient descent to try and force the module to break its constraints. This “adversarial attack” verifies that even under optimization, the module enforces its constraints.

To run the adversarial tests:

```bash
pytest -v priority_approach/test_adversarial.py
```

## Project Structure

```
.
├── priority_approach/
│   ├── constraints.py        # Implementation of constraint modules.
│   ├── test_constraints.py   # Unit tests for constraint modules.
│   ├── test_adversarial.py   # Adversarial tests attacking the constraints.
│   ├── __init__.py           # (Optional) Enables package-style imports.
│   └── requirements.txt      # (Optional) List of required Python packages.
├── README.md                 # This read-me file.
└── .gitignore                # Git ignore file.
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements, additional features, or bug fixes.

## License

[MIT License](LICENSE)
