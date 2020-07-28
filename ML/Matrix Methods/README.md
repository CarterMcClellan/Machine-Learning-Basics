## Matrix Methods
Matrix Methods (e.g SVD, QR, Eigenvalue) get used in OLS, Basis approximations, dimensionality reduction problems, etc... All sorts of important Machine Learning computations can be accelerated by some clever new formulation. Finding the projection matrix or the pseudo-inverse are both problems which benefit from SVD, as an example.

![formula](https://render.githubusercontent.com/render/math?math=A%20=%20U%20\Sigma%20V^T%20\implies%20A^{-1}%20=%20U%20\Sigma^{-1}V^{T})


But all this stuff is pretty rigorous, and inaccesible at a university level, so I thought I would build a couple different decompositions in numpy following the textbooks, then mess around with them and see if I could get a better handle on them.

|                                    | nb viewer | tex | complete           |
|------------------------------------|-----------|-----|--------------------|
| QR Decomposition                   |           |     | :heavy_check_mark: |
| Jordan Decomposition               |           |     | :x:                |
| Singular Value Decomposition (SVD) |           |     | :x:                |
| Eigen Value Decomposition          |           |     | TODO               |
| PolyVandermode Matrix              |           |     | TODO               |

### Useful Links
- [Numerical Analysis @ Cornell](https://www.cs.cornell.edu/~bindel/class/cs6210-f12/lectures.html)
- [Numerical Analysis @ UToronto](https://www.cs.toronto.edu/~lczhang/csc338_20191/homework.html)
- [Numerical Analysis @ WI](https://www.math.wisc.edu/~jroos/19.2.514/index.html)
