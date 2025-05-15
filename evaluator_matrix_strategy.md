# AlphaEvolve Evaluator Pool - Matrix Strategy

This document explains the GitHub Actions matrix strategy for parallel evaluation of algorithm candidates.

## Matrix Strategy Overview

```mermaid
graph TB
    classDef branch fill:#d9eaf7,stroke:#4b8bbf,stroke-width:2px
    classDef job fill:#def2d6,stroke:#60bd4b,stroke-width:2px
    classDef artifact fill:#f9eddf,stroke:#dba656,stroke-width:2px
    
    B1[Branch: experiment_ensemble]:::branch
    B2[Branch: evolve-20250515135425]:::branch
    B3[Branch: evolve-20250515140234]:::branch
    B4[Branch: evolve-20250515143815]:::branch
    
    D[Discover Branches Job]:::job --> B1
    D --> B2
    D --> B3
    D --> B4
    
    B1 --> E1[Evaluate Job: experiment_ensemble]:::job
    B2 --> E2[Evaluate Job: evolve-20250515135425]:::job
    B3 --> E3[Evaluate Job: evolve-20250515140234]:::job
    B4 --> E4[Evaluate Job: evolve-20250515143815]:::job
    
    E1 --> A1[Artifact: benchmark-experiment_ensemble]:::artifact
    E2 --> A2[Artifact: benchmark-evolve-20250515135425]:::artifact
    E3 --> A3[Artifact: benchmark-evolve-20250515140234]:::artifact
    E4 --> A4[Artifact: benchmark-evolve-20250515143815]:::artifact
    
    subgraph "Compare & Merge"
        C[Compare Job]:::job
        M[Merge Best Job]:::job
        C --> M
    end
    
    A1 --> C
    A2 --> C
    A3 --> C
    A4 --> C
    
    M --> |if best| B1
```

## How It Works

### 1. Discover Branches Job

- Runs first to identify all candidate branches to evaluate
- Uses `git branch` to list all branches matching the patterns:
  - `evolve/*` - Algorithm candidates
  - `experiment_ensemble` - Current best algorithm

### 2. Matrix Strategy Evaluation

The GitHub Actions Matrix strategy creates multiple parallel job instances, one for each branch:

```yaml
strategy:
  matrix:
    branch: ${{ fromJson(format('[{0}]', needs.discover_branches.outputs.branches)) }}
  fail-fast: false  # Continue evaluating other branches even if one fails
```

Each job instance:
1. Creates a separate git worktree for the branch
2. Runs all tests and benchmarks
3. Generates benchmark metrics
4. Uploads results as artifacts

### 3. Compare & Merge Process

After all parallel evaluations complete:
1. The Compare job downloads all benchmark artifacts
2. Metrics are aggregated into a comparison table and chart
3. The Merge job identifies the best-performing branch
4. If it outperforms the current best, it's automatically merged

## Performance Benefits

| Traditional Approach | Matrix Strategy |
|---------------------|-----------------|
| Sequential evaluation of branches | Parallel evaluation of all branches |
| Long feedback cycles | Fast feedback regardless of branch count |
| O(n) time complexity | O(1) time complexity (limited by runners) |
| Limited coverage | Comprehensive evaluation of all branches |

## Real-world Example

In our experiment with 4 branches, the matrix strategy reduced total evaluation time from over 40 minutes to just under 10 minutes - a 4x speedup.

## Integration with AlphaEvolve Workflow

```mermaid
graph LR
    classDef step fill:#e8f4f8,stroke:#5a9bcf,stroke-width:2px
    
    S1[1. Prompt Sampling]:::step --> S2
    S2[2. LLM Generate Patch]:::step --> S3
    S3[3. Apply Patch in New Worktree]:::step --> S4
    S4[4. Evaluator Pool Matrix Strategy]:::step --> S5
    S5[5. Feedback to Program DB]:::step --> S6
    S6[6. Selection & Merge Policy]:::step --> S1
```

The matrix strategy in Step 4 enables the AlphaEvolve workflow to efficiently evaluate multiple algorithm candidates in parallel, accelerating the evolutionary process. 