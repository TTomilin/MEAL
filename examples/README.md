# MEAL Examples

This directory contains examples demonstrating how to use the MEAL (Multi-Agent Environment for Continual Learning) library with its gym-style API.

## Examples Overview

### 1. `basic_usage.py`
**Purpose**: Demonstrates the basic gym-style API for creating and interacting with MEAL environments.

**Key Features**:
- Using `meal.make_env()` and `meal.make()` functions
- Listing available environments with `meal.list_envs()`
- Basic environment interaction (reset, step, action sampling)

**Usage**:
```bash
python examples/basic_usage.py
```

### 2. `continual_learning_sequence.py`
**Purpose**: Shows how to generate and use sequences of environments for continual learning scenarios.

**Key Features**:
- Using `meal.make_cl_sequence()` to generate environment sequences
- Different strategies: 'curriculum', 'random', 'generate'
- Environment metadata tracking (task ID, name, strategy)
- Simple evaluation across multiple environments

**Usage**:
```bash
python examples/continual_learning_sequence.py
```

### 3. `environment_types.py`
**Purpose**: Demonstrates the different types of environments available in MEAL and their configurations.

**Key Features**:
- Standard Overcooked environment
- Partially Observable Overcooked (`overcooked_po`)
- N-Agent Overcooked with different agent counts
- Environment information display
- Interactive environment demonstration

**Usage**:
```bash
python examples/environment_types.py
```

## Quick Start

To get started with MEAL, try the basic usage example:

```python
import meal

# List available environments
print(meal.list_envs())

# Create an environment
env = meal.make_env('overcooked')

# Generate a continual learning sequence
cl_envs = meal.make_cl_sequence(sequence_length=5, strategy='curriculum')
```

## API Reference

### Core Functions

- `meal.make_env(env_id, **kwargs)`: Create a single environment (gym-style)
- `meal.make(env_id, **kwargs)`: Create a single environment (original API)
- `meal.make_cl_sequence(...)`: Generate a sequence of environments for continual learning
- `meal.list_envs()`: List all available environment IDs

### Continual Learning Strategies

- **'curriculum'**: Environments with increasing difficulty (easy → medium → hard)
- **'random'**: Random selection from available layouts (no immediate repeats)
- **'ordered'**: Deterministic sequence through available layouts
- **'generate'**: Procedurally generated environments on-the-fly

### Environment Types

- **'overcooked'**: Standard 2-agent Overcooked environment
- **'overcooked_po'**: Partially observable Overcooked
- **'overcooked_n_agent'**: N-agent Overcooked (specify `num_agents`)

## Requirements

Make sure you have MEAL installed:

```bash
pip install -e .
```

For visualization examples (if any), also install:

```bash
pip install -e ".[viz]"
```

## Notes

- All examples use JAX for random number generation
- Environments are JAX-compatible and support vectorization
- The continual learning API automatically adds metadata to environments for tracking