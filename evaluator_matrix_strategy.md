# GitHub Actions Matrix Strategy for AlphaEvolve Evaluators

This document describes how we implement Step 4 (Evaluator Pool) of the AlphaEvolve workflow using GitHub Actions matrix strategy to parallelize algorithm candidate evaluations.

## Overview

The AlphaEvolve workflow is designed to iteratively improve machine learning algorithms through an automated process of generation, evaluation, and selection. A critical component of this workflow is the **Evaluator Pool**, which tests multiple algorithm candidates to determine which ones should be promoted to the next generation.

By leveraging GitHub Actions matrix strategy, we can:

1. Evaluate multiple algorithm candidates in parallel
2. Standardize the evaluation criteria across all candidates
3. Automatically collect performance metrics
4. Make data-driven decisions about which algorithms to keep or discard

## Implementation Architecture

![AlphaEvolve Matrix Strategy](https://mermaid.ink/img/pako:eNqFU8tu2zAQ_BWCpxYoDPvVxLl1ESB9oO2hKJIeGGlpC5FIlaQSG4b_vUuJkhO7aXThcmdnZ2d3l1lmCmSJeSNbhRoq3YIWWv0GbR8gU5Krb5AKQadWrxpqlUljN2oLOT3cVVu9AcnU_vIKMlO_oR0bnUY1GGfhAFMjqY0WOWrVmBqO0r9v6aMUcLQCx4Y0_7G1gYvFAqSWNdLqPy6hf3t-X4FU04PaNNTxWzpnqTlK8xDO7mhpSOG9UbVSVOeVR_7FcX7vA3lbK7mHLMeZ-oGQWMIOdTAUwTmqcJDLtgpdjGEaIFLXLkCZ1a7Bh3jLN7T9Q9pVzp7T0KOoS0JuhqKXLKwNlbzHCuLOtUVXUeWvMI-F4C1lCq2VxAYKu8Cjl95cKBQnwTH2B4_tnWqKQiHNh9RoaVzMHSGFGscCz4eRiKVlx6qcubbT_MCZnnkcT0Eh4y6eR4zTxL4cwLlvGDMSOsQ9OQMzJbkjLUePTIhxFLuM96fPcDu-yB2_vbjE5m29plWUvvw3XZnlg3S5drgkVfQpnYDSGfcJ3wfMK_rwuJ6nFPgzBj2j3G9mfzGbQTmDeYpbj_I3SycY4kxYxS7wfoPDEbqJhUKzPZhHVL8a7ojtCqVIVR4b8e46Udk2UEZoLCzRu9w4snZQS1PAZ0yS4nJ5k-Z_wE0HsaRLsNj8GyqKSfENfsSZrXQ9pT_Lkni-JUuMkXzMkjjlk9uZ38ZTMZ0kSRQVSRiVYZJPyjhOiulsk6YLHvGfWZLwFHKO47Q0LSYrNkmTIkzScDa9cV_fzELTmmTuTz6PPg9feGGStCaLfwDnmjwb)

### Workflow Components

Our implementation consists of two primary GitHub Actions workflows:

1. **`evaluate_candidates.yml`** - Runs the evaluator pool in parallel
2. **`merge_best_candidate.yml`** - Analyzes results and merges the best candidate

### Matrix Strategy for Parallel Evaluation

The matrix strategy allows us to spawn multiple GitHub Actions runners, each evaluating a different algorithm candidate. Here's how it works:

```yaml
strategy:
  matrix:
    branch: ${{ fromJson(needs.detect-branches.outputs.branches) }}
  fail-fast: false
```

This creates a separate job for each branch in our matrix, allowing all candidates to be evaluated simultaneously rather than sequentially.

## The Evaluation Process

For each algorithm candidate (branch), the evaluation follows these steps:

1. **Setup Environment** - Clone repository and install dependencies
2. **Create Worktree** - Create a git worktree for the candidate branch
3. **Run Benchmarks** - Execute performance benchmarks
4. **Run Tests** - Validate algorithm correctness
5. **Generate Reports** - Create comprehensive benchmark reports
6. **Store Results** - Save metrics as artifacts and in git notes

Each step is executed in parallel across all algorithm candidates, dramatically reducing the total evaluation time.

## Metrics Collection and Storage

We collect various performance metrics:
- Accuracy, precision, recall, F1 score
- Training and inference time
- Memory usage and CPU utilization
- Component-level performance benchmarks

These metrics are stored in:
1. **JSON files** - Structured data for programmatic analysis
2. **Markdown reports** - Human-readable benchmark reports
3. **Git notes** - Attaching metadata to commits without changing the code
4. **Workflow artifacts** - Persistent storage of all benchmark data

## Selection and Merge Process

After evaluation, the `merge_best_candidate.yml` workflow:

1. Downloads all evaluation artifacts
2. Compares metrics across all candidates
3. Identifies the best-performing algorithm
4. Creates a PR or auto-merges the winning branch

The selection process uses a configurable scoring system that weights different metrics according to their importance:

```python
weights = {
    'accuracy': 1.0,
    'f1_score': 0.8,
    'training_time': -0.3,  # Negative weight (lower is better)
    'inference_time': -0.5,
    'memory_usage': -0.3,
}
```

## Advantages Over Sequential Evaluation

The matrix strategy provides several key advantages:

1. **Parallel Execution** - All candidates are evaluated simultaneously
2. **Consistent Environment** - Each candidate is tested in the same environment
3. **Scalability** - Easy to add new metrics or test dimensions
4. **Fully Automated** - No human intervention required for evaluation and selection

## Example Workflow Execution

When a new branch with prefix `evolve/` is pushed:

1. The `evaluate_candidates.yml` workflow triggers
2. It detects all `evolve/*` branches
3. Creates a matrix job for each branch
4. Each job sets up a worktree and runs benchmarks
5. Results are saved as artifacts
6. The `merge_best_candidate.yml` workflow is triggered
7. It analyzes the results and merges the best performer

## Implementation in AlphaEvolve

This matrix evaluation strategy forms a critical part of the AlphaEvolve closed-loop system:

```
Prompt Sampler → LLM Generate Patch → Apply in Worktree → Evaluator Pool (Matrix) → Feedback → Selection
```

By parallelizing the evaluation step, we dramatically accelerate the evolution process, allowing more algorithm candidates to be tested in less time, increasing the chances of finding an optimal solution.

## Conclusion

The GitHub Actions matrix strategy enables us to implement a robust, parallel evaluation system for the AlphaEvolve workflow. This approach significantly speeds up the algorithm evolution process by allowing multiple candidates to be evaluated simultaneously, greatly reducing the time required for each iteration of the evolutionary process. 