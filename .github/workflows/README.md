# AlphaEvolve Evaluator Pool - GitHub Actions

This directory contains GitHub Actions workflows for implementing the "Evaluator Pool" component of our AlphaEvolve workflow. The evaluator pool is responsible for:

1. Running multiple algorithm candidates in parallel
2. Collecting performance metrics for each candidate
3. Comparing results to identify the best algorithm

## How It Works

The `evaluate_candidates.yml` workflow implements a matrix strategy to:

1. Automatically detect branches with `evolve/` prefix
2. Create a separate git worktree for each branch
3. Run all tests and benchmarks in parallel
4. Collect and compare performance metrics
5. Identify the best-performing branch

## Workflow Structure

The workflow is organized in three jobs:

### 1. Discover Branches

- Detects all branches matching the pattern (`evolve/*` and `experiment_ensemble`)
- Outputs a list of branches to evaluate

### 2. Evaluate

- Runs as a matrix job, with one instance per branch
- Sets up a separate git worktree for each branch
- Runs unit tests and multiple types of benchmarks:
  - Overall benchmark (execution time, memory, CPU usage)
  - Component-level benchmarks (feature engineering, model training, prediction)
  - pytest benchmarks for detailed timing
- Generates benchmark reports and metrics
- Records performance metrics using git notes

### 3. Compare Branches

- Downloads all benchmark results
- Generates comparison tables and charts
- Identifies the best algorithm based on accuracy and performance
- Outputs a comprehensive comparison report

## Matrix Strategy Benefits

Using GitHub Actions' matrix strategy gives us several advantages:

1. **Parallel Execution**: All branches are evaluated simultaneously
2. **Isolation**: Each branch runs in its own worktree environment
3. **Scalability**: Easy to add new dimensions to the matrix (e.g., Python versions, datasets)
4. **Fault Tolerance**: One failing branch doesn't stop the evaluation of others

## Integration with AlphaEvolve Workflow

This Evaluator Pool fits into the broader AlphaEvolve workflow:

1. **Step 1 (Prompt Sampling)** gathers information about previous attempts
2. **Step 2 (LLM Generate Patch)** creates algorithm improvements
3. **Step 3 (Apply Patch in New Worktree)** creates a new branch with changes
4. **Step 4 (Evaluator Pool)** 👈 **You are here** - evaluates all branches in parallel
5. **Step 5 (Feedback to Program DB)** stores metrics as git notes
6. **Step 6 (Selection & Merge Policy)** identifies and merges the best algorithm

## Viewing Results

After the workflow runs:

1. Check the "Actions" tab in GitHub
2. Find the most recent "Algorithm Evaluator Pool" run
3. Download artifacts:
   - `benchmark-{branch}` - individual branch results
   - `branch-comparison` - overall comparison report 