# BrownianMotionTree

A package to compute the maximum likelihood of brownian motion tree models.

## Example

```julia
using BrownianMotionTree

# Take the 4-star tree
T = [1 1 0 0 0
     1 0 1 0 0
     1 0 0 1 0
     1 0 0 0 1]

# Compute the MLE for a generic instance
mle = MLE(T)

# sample a random positive definite matrix
S = rand_pos_def(4)

# Compute the criticical points
crits = solve(mle, S)

# we can also explore the boundary strata
boundary_crits = star_tree_boundary(T, S)
```
