# Sparse-PCA-with-Elastic-Net
Python implementation of Sparse PCA proposed by Hui Zou, Trevor Hastie and Robert Tibshirani. This implementation has been tested by Pitprops data.

This method can generate sparse loadings (with the help of elastic net algorithm). The elastic net algorithm I used here is from scikit learn: `sklearn.linear_models.ElasticNet`.

## Objective / Loss Function

![of](https://latex.codecogs.com/svg.latex?%5Cmathbf%7BA%7D%2C%5Cmathbf%7BB%7D%3D%20%5Cmathop%7B%5Carg%5Cmin%7D_%7B%5Cmathbf%7BA%7D%2C%5Cmathbf%7BB%7D%7D%5Cleft%5C%7C%5Cmathbf%7BX%7D-%5Cmathbf%7BXB%7D%5Cmathbf%7BA%7D%5ET%5Cright%5C%7C%5E2_F%20&plus;%5Clambda%5Csum_%7Bi%3D1%7D%5Ek%5Cleft%5C%7C%5Cboldsymbol%7B%5Cbeta%7D_i%5Cright%5C%7C%5E2_2%20&plus;%5Csum_%7Bi%3D1%7D%5Ek%5Clambda_%7B1%2Cj%7D%5Cleft%5C%7C%5Cboldsymbol%7B%5Cbeta%7D_i%5Cright%5C%7C_1)

with A subject to 

![st](https://latex.codecogs.com/svg.latex?%5Cmathbf%7BA%7D%5ET%5Cmathbf%7BA%7D%20%3D%20%5Cmathbf%7BI%7D_%7Bk%20%5Ctimes%20k%7D)

## Algorithm
According to the original paper, A and B are optimized alternatively while the other one is fixed.

**B fixed:** Reduced Rank Procrustes Rotation method is used. In this situation A has a closed form solution that can be calculated directly by formula.

**A fixed:** Optimizing B here can be seen as optimizing an elastic net problem. Using sci-kit learning tools, this can be achieved.

## Test on pitprops data
The data is tested on pitprops data, and generated almost the same results shown in the paper.

## Reference
- Zou, H., et al. (2006). "Sparse Principal Component Analysis." Journal of Computational and Graphical Statistics 15(2): 265-286.
