
### **What is NumPy?**
[Docs](https://numpy.org/doc/stable/user/index.html)
**NumPy** (Numerical Python) is a fundamental Python library for **scientific computing**. It provides:

1. **Powerful N-dimensional arrays (`ndarray`)**

   * Efficient storage and operations on large datasets.
   * Supports 1D, 2D, 3D, … arrays (vectors, matrices, tensors).

2. **Mathematical functions**

   * Fast element-wise operations, linear algebra, statistics, and more.
   * Examples: `np.dot`, `np.matmul`, `np.sum`, `np.mean`.

3. **Broadcasting**

   * Allows operations on arrays of different shapes without explicit loops.

4. **Integration with other libraries**

   * Many libraries (Pandas, SciPy, Matplotlib, TensorFlow) are built on NumPy arrays.

5. **Performance**

   * Written in C, so much faster than native Python loops for numerical tasks.

---

### **Typical Use-Cases**

* Array manipulation and reshaping
* Linear algebra and matrix operations
* Statistics and data analysis
* Signal processing or image manipulation
* Machine learning and scientific simulations

---

That’s a big request — NumPy has *hundreds* of functions and methods. I can’t list *every* single one in detail here, but I can cover a fairly comprehensive set: many of the most used functions, categories, their explanations, and good examples. If you like, I can then send you a downloadable reference (or split into multiple parts). Let’s go through major categories, with explanations and examples.

---

## What is NumPy?

NumPy (“Numerical Python”) is the fundamental package for numerical computation in Python. Core to it is the **ndarray**, an N-dimensional array object. The library provides:

* array creation routines
* array manipulation / reshaping / indexing / slicing
* mathematical operations (elementwise, linear algebra)
* statistical, logical, Fourier transform, etc.

---

## Array Creation

| Function                                                  | What it does / Use-case                                                                                                      | Example                                                                          |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `np.array(object, dtype=None, ...)`                       | Create an ndarray from Python list/tuple (or sequence or nested sequences). You can specify data type.                       | `np.array([1, 2, 3])` → `array([1,2,3])`; `np.array([[1,2],[3,4]], dtype=float)` |
| `np.zeros(shape, dtype=float, ...)`                       | Make an array filled with zeros, of given shape. Useful for initialization.                                                  | `np.zeros((3,4))` → 3×4 matrix of zeros                                          |
| `np.ones(shape, dtype, ...)`                              | Same, but filled with ones.                                                                                                  | `np.ones(5)` → `[1,1,1,1,1]`                                                     |
| `np.empty(shape, ...)`                                    | Create array without initializing entries (so contents are whatever is in memory). Use when you’ll fill it soon (for speed). | `np.empty((2,2))` might give random values until filled                          |
| `np.full(shape, fill_value, ...)`                         | Create array filled with a given value.                                                                                      | `np.full((3,3), 7)` → all entries are 7                                          |
| `np.eye(n, m=None, k=0, ...)` / `np.identity(n)`          | Identity matrix (1’s on diagonal, 0 elsewhere). Useful in linear algebra.                                                    | `np.eye(3)` → `[[1,0,0],[0,1,0],[0,0,1]]`                                        |
| `np.arange(start, stop, step, dtype)`                     | Like Python’s `range`, but returns array. Useful for generating sequences.                                                   | `np.arange(0,10,2)` → `[0,2,4,6,8]`                                              |
| `np.linspace(start, stop, num, endpoint=True, ...)`       | Generate `num` evenly spaced samples between `start` and `stop`. Useful for plotting, sampling.                              | `np.linspace(0,1,5)` → `[0. ,0.25,0.5,0.75,1.0]`                                 |
| `np.logspace(start, stop, num, base=10.0, ...)`           | Samples are in log scale: powers of `base`.                                                                                  | `np.logspace(0,2,5)` → `[1, 3.1623, 10, 31.623, 100]`                            |
| `np.random.rand(d0, d1, ..., dn)`                         | Uniformly distributed random numbers in `[0,1)`.                                                                             | `np.random.rand(2,3)` → 2×3 array of uniform randoms                             |
| `np.random.randn(d0, d1, ..., dn)`                        | Standard normal (Gaussian) distribution randoms.                                                                             | `np.random.randn(3,3)`                                                           |
| `np.random.randint(low, high=None, size=None, dtype=int)` | Random integers in `[low, high)` or `[0, high)` if `high` is None.                                                           | `np.random.randint(0, 10, size=(2,2))`                                           |
| `np.zeros_like(a)` / `np.ones_like(a)`                    | Create new array of same shape as `a`, filled with zeros or ones.                                                            | `a = np.array([[1,2],[3,4]])`; `np.zeros_like(a)` → same shape, zeros            |

---

## Array Inspection / Properties

| Function / attribute       | What it gives                                                                                                                                | Example                                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `a.shape`                  | Tuple of dimensions.                                                                                                                         | `a = np.array([[1,2,3],[4,5,6]])`; `a.shape` → `(2,3)`                                                    |
| `a.ndim`                   | Number of axes (dimensions).                                                                                                                 | same `a`, `a.ndim` → `2`                                                                                  |
| `a.dtype`                  | Data type of elements.                                                                                                                       | `np.array([1,2,3]).dtype` → `int64`; `np.array([1.2, 3.4]).dtype` → `float64`                             |
| `a.size`                   | Total number of elements (product of shape dims).                                                                                            | `(2,3)` → `6`                                                                                             |
| `a.itemsize`               | Number of bytes per element.                                                                                                                 | `np.array([1,2,3], dtype=np.int32).itemsize` → `4`; `np.array([1.2], dtype=np.float64).itemsize` → `8`    |
| `a.nbytes`                 | Total bytes consumed = `itemsize * size`.                                                                                                    | `np.array([1,2,3], dtype=np.int32).nbytes` → `12`                                                         |
| `np.info(obj)`             | Print documentation/info about obj (function, class, etc.)                                                                                   | `np.info(np.arange)` prints info about the `arange` function                                              |
| `np.set_printoptions(...)` | Control how arrays are printed (precision, threshold, suppress small values, etc.). Example: suppress scientific notation or small decimals. | `np.set_printoptions(precision=3, suppress=True); print(np.array([1.23456e-10, 1.23456]))` → `[0. 1.235]` |

---

## Reshaping, Transposing, Repeating, Tiling, etc.

| Function / Method                     | What it does / When useful                                                                                                         | Example                                                                                                          |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `a.reshape(new_shape)`                | Returns a view (if possible) of `a` with a different shape, same data. Number of elements must match. Useful to change dimensions. | `a = np.arange(6)` → `[0,1,2,3,4,5]`; `a.reshape((2,3))` → `[[0,1,2],[3,4,5]]`                                   |
| `a.T` or `np.transpose(a, axes=None)` | Swap axes / transpose: flip rows & columns for 2D; more general for n-D.                                                           | `b = np.array([[1,2,3],[4,5,6]])`; `b.T` → `[[1,4],[2,5],[3,6]]`                                                 |
| `np.repeat(a, repeats, axis=None)`    | Repeat elements of array. If `axis` given, across that dimension. Useful for duplicating data.                                     | `np.repeat([1,2,3], 2)` → `[1,1,2,2,3,3]`<br>`np.repeat([[1,2],[3,4]], 2, axis=0)` → `[[1,2],[1,2],[3,4],[3,4]]` |
| `np.tile(a, reps)`                    | Tile: repeat the whole array in block fashion.                                                                                     | `np.tile([1,2], 3)` → `[1,2,1,2,1,2]`<br>`np.tile([[1,2],[3,4]], (2,3))` → repeats in both dims                  |

---

## Elementwise / Universal Functions (ufuncs)

These apply operations to each element (or pair, etc.) without explicit loops.

| Function                              | What it does                                                                                  | Example                                                                       |
| ------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `np.add(x, y)` or `x + y`             | Elementwise addition.                                                                         | `np.add([1,2,3],[4,5,6])` → `[5,7,9]`                                         |
| `np.subtract(x, y)` or `x - y`        | Elementwise subtraction.                                                                      | `np.subtract([5,7,9],[1,2,3])` → `[4,5,6]`                                    |
| `np.multiply(x, y)` or `x * y`        | Elementwise multiplication.                                                                   | `np.multiply([1,2,3],[4,5,6])` → `[4,10,18]`                                  |
| `np.divide(x, y)` or `x / y`          | Elementwise division.                                                                         | `np.divide([10,20,30],[2,5,10])` → `[5.,4.,3.]`                               |
| `np.power(x, y)` or `x**y`            | Raise elements.                                                                               | `np.power([2,3,4],[3,2,1])` → `[8,9,4]`                                       |
| `np.sqrt(x)`                          | Square root elementwise.                                                                      | `np.sqrt([1,4,9])` → `[1.,2.,3.]`                                             |
| `np.exp(x)`                           | Exponential \$e^x\$ on each element.                                                          | `np.exp([0,1,2])` → `[1.,2.718...,7.389...]`                                  |
| `np.log(x)`                           | Natural logarithm.                                                                            | `np.log([1, np.e, np.e**2])` → `[0.,1.,2.]`                                   |
| `np.log10(x)`, `np.log2(x)`           | Logs in base 10 or 2.                                                                         | `np.log10([1,10,100])` → `[0.,1.,2.]`; `np.log2([1,2,4,8])` → `[0.,1.,2.,3.]` |
| `np.sin(x)`, `np.cos(x)`, `np.tan(x)` | Trigonometric functions.                                                                      | `np.sin([0, np.pi/2])` → `[0.,1.]`; `np.cos([0, np.pi])` → `[1.,-1.]`         |
| `np.arcsin(x)`, etc.                  | Inverse trigonometric.                                                                        | `np.arcsin([0,1])` → `[0.,1.5708]` (radians)                                  |
| `np.sinh(x)`, `np.cosh(x)`, …         | Hyperbolic.                                                                                   | `np.cosh([0,1])` → `[1.,1.543...]`                                            |
| `np.abs(x)` or `np.absolute(x)`       | Absolute value.                                                                               | `np.abs([-3,-1,0,2])` → `[3,1,0,2]`                                           |
| `np.sign(x)`                          | Returns elementwise −1, 0, +1 depending on negative, zero, positive.                          | `np.sign([-5,0,7])` → `[-1,0,1]`                                              |
| `np.clip(x, min, max)`                | Limit values to a range. Values below min set to min, above max to max. Useful to bound data. | `np.clip([1,5,10], 2, 8)` → `[2,5,8]`                                         |

---

## Aggregation / Reduction / Statistics

| Function                                                  | What it does                                                    | Example                                                                         |
| --------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `np.sum(a, axis=None, ...)`                               | Sum of elements. Axis specifies dimension(s) over which to sum. | `np.sum([[1,2],[3,4]])` → `10`; axis=0 → `[4,6]`; axis=1 → `[3,7]`              |
| `np.prod(a, axis=None, …)`                                | Product over elements.                                          | `np.prod([1,2,3,4])` → `24`                                                     |
| `np.mean(a, axis=None, …)`                                | Arithmetic mean.                                                | `np.mean([1,2,3,4])` → `2.5`                                                    |
| `np.std(a, axis=None, ddof=0, …)`                         | Standard deviation.                                             | `np.std([1,2,3,4])` → `1.118…`                                                  |
| `np.var(a, axis=None, …)`                                 | Variance.                                                       | `np.var([1,2,3,4])` → `1.25`                                                    |
| `np.min(a, axis=None, …)` / `np.maximum.reduce(...)` etc. | Minimum value.                                                  | `np.min([3,1,4,2])` → `1`; `np.maximum.reduce([3,1,4,2])` → `4`                 |
| `np.max(a, axis=None, …)`                                 | Maximum value.                                                  | `np.max([3,1,4,2])` → `4`                                                       |
| `np.argmin(a, axis=None)` / `np.argmax(a, axis=None)`     | Indices of minimum / maximum elements.                          | `np.argmin([3,1,4,2])` → `1`; `np.argmax([3,1,4,2])` → `2`                      |
| `np.median(a, axis=None)`                                 | The median value.                                               | `np.median([1,3,2,4])` → `2.5`                                                  |
| `np.percentile(a, q, axis=None)`                          | The q-th percentile (e.g. 25th, 50th).                          | `np.percentile([1,2,3,4], 50)` → `2.5`; `np.percentile([1,2,3,4], 25)` → `1.75` |
| `np.any(a, axis=None)`                                    | Is any element in array True (or nonzero)?                      | `np.any([0,0,1])` → `True`; `np.any([0,0,0])` → `False`                         |
| `np.all(a, axis=None)`                                    | Are all elements True (or nonzero)?                             | `np.all([1,2,3])` → `True`; `np.all([1,0,3])` → `False`                         |

---

## Linear Algebra

| Function                                  | What it does / Use-case                                                                                                                                      | Example                                                                                           |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `np.dot(a, b)`                            | Dot product: for 1D vectors, scalar; for 2D, matrix multiplication; more general with contraction of last axis of `a` and second-last of `b` in higher dims. | `np.dot([1,2,3], [4,5,6])` → `32`<br>`np.dot([[1,2],[3,4]], [[5,6],[7,8]])` → `[[19,22],[43,50]]` |
| `np.matmul(a, b)`                         | Matrix product; similar to dot but handles stacking / broadcasted matmul differently. For many use cases, identical to dot.                                  | `np.matmul([[1,2],[3,4]], [[5],[6]])` → `[[17],[39]]`                                             |
| `np.linalg.inv(a)`                        | Inverse of square matrix. Useful in solving linear systems, etc.                                                                                             | `np.linalg.inv([[1,2],[3,4]])` → `[[-2. , 1. ], [1.5, -0.5]]`                                     |
| `np.linalg.det(a)`                        | Determinant of matrix.                                                                                                                                       | `np.linalg.det([[1,2],[3,4]])` → `-2.0`                                                           |
| `np.linalg.eig(a)`                        | Eigenvalues and eigenvectors.                                                                                                                                | `vals, vecs = np.linalg.eig([[2,0],[0,3]])` → `vals=[2,3]`, `vecs=[[1,0],[0,1]]`                  |
| `np.linalg.solve(a, b)`                   | Solve linear system `a x = b`.                                                                                                                               | `np.linalg.solve([[3,1],[1,2]], [9,8])` → `[2., 3.]`                                              |
| `np.linalg.norm(a, ord=None, axis=None)`  | Norm (length) of vectors / matrices.                                                                                                                         | `np.linalg.norm([3,4])` → `5.0`                                                                   |
| `np.trace(a, offset=0, axis1=0, axis2=1)` | Sum of diagonal elements.                                                                                                                                    | `np.trace([[1,2],[3,4]])` → `5`                                                                   |
| `np.linalg.svd(a)`                        | Singular Value Decomposition.                                                                                                                                | `U, S, Vt = np.linalg.svd([[1,2],[3,4]])`                                                         |
| `np.linalg.pinv(a)`                       | Pseudoinverse (for non-square or singular matrices).                                                                                                         | `np.linalg.pinv([[1,2],[3,4],[5,6]])`                                                             |
| `np.transpose` (also in earlier section)  | Change axes / transpose matrix.                                                                                                                              | `np.transpose([[1,2,3],[4,5,6]])` → `[[1,4],[2,5],[3,6]]`                                         |

---

## Rounding, Comparisons, Logical

| Function                                                                         | What it does                                       | Example                                                                                   |
| -------------------------------------------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `np.round(a, decimals=0)`                                                        | Round to given decimal places.                     | `np.round([1.234, 2.345], 1)` → `[1.2, 2.3]`                                              |
| `np.floor(a)` / `np.ceil(a)`                                                     | Round down / up to nearest integer.                | `np.floor([1.7, -1.2])` → `[1., -2.]`<br>`np.ceil([1.2])` → `[2.]`                        |
| `np.trunc(a)`                                                                    | Truncate decimal part.                             | `np.trunc([-1.7, 1.7])` → `[-1., 1.]`                                                     |
| `np.modf(a)`                                                                     | Return fractional and integer parts.               | `np.modf([2.3, -2.3])` → `([0.3, -0.3], [2., -2.])`                                       |
| `np.isnan(a)` / `np.isfinite(a)` / `np.isinf(a)`                                 | Check for NaNs, infinite, etc.                     | `np.isnan([1, np.nan])` → `[False, True]`<br>`np.isfinite([1, np.inf])` → `[True, False]` |
| Comparison operators / functions: `np.equal`, `np.not_equal`, `np.greater`, etc. | Elementwise comparisons returning boolean arrays.  | `np.greater([2, 4], [1, 4])` → `[True, False]`                                            |
| Logical: `np.logical_and`, `np.logical_or`, `np.logical_not`, `np.where`         | Useful for masking / branching based on condition. | `np.where([True, False], [1, 2], [3, 4])` → `[1, 4]`                                      |

---

## Utility / Miscellaneous

| Function / Method                                                          | What it does                                                             | Example                                                                                                                                                                                       |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `np.reshape`, `np.ravel`                                                   | `ravel` returns flattened **view** if possible; `reshape` changes shape. | `python import numpy as np a = np.arange(6).reshape(2,3) # [[0,1,2],[3,4,5]] np.reshape(a,(3,2))  # → [[0,1],[2,3],[4,5]] a.ravel()  # → [0,1,2,3,4,5] `                                      |
| `np.squeeze(a)` / `np.expand_dims(a, axis)`                                | Remove axes of length 1; add new axes. Useful to adjust shapes.          | `python a = np.array([[[1],[2],[3]]])  # shape (1,3,1) np.squeeze(a).shape      # → (3,) b = np.expand_dims(a, axis=0) b.shape                # → (1,1,3,1) `                                 |
| `np.concatenate(arrays, axis=0)`                                           | Join multiple arrays along an axis.                                      | `python a = np.array([1,2,3]) b = np.array([4,5,6]) np.concatenate([a,b])   # → [1,2,3,4,5,6] `                                                                                               |
| `np.stack(arrays, axis=0)`                                                 | Join arrays along a new axis.                                            | `python a = np.array([1,2,3]) b = np.array([4,5,6]) np.stack([a,b], axis=0) # → [[1,2,3],[4,5,6]] np.stack([a,b], axis=1) # → [[1,4],[2,5],[3,6]] `                                           |
| `np.hstack`, `np.vstack`, `np.dstack`                                      | Stack arrays horizontally, vertically, or depth-wise.                    | `python a = np.array([1,2,3]) b = np.array([4,5,6]) np.hstack([a,b]) # → [1,2,3,4,5,6] np.vstack([a,b]) # → [[1,2,3],[4,5,6]] np.dstack([a,b]) # → [[[1,4],[2,5],[3,6]]] `                    |
| `np.split(a, indices_or_sections, axis=0)` / `np.hsplit`, `np.vsplit` etc. | Split arrays into parts.                                                 | `python a = np.arange(9).reshape(3,3) # [[0,1,2],[3,4,5],[6,7,8]] np.split(a,3)   # 3 arrays: each row np.hsplit(a,3)   # 3 arrays: each column np.vsplit(a,3)   # 3 arrays: each row again ` |
| `np.copy(a)`                                                               | Make a **deep copy** (independent).                                      | `python a = np.array([1,2,3]) b = np.copy(a) b[0] = 99 a  # → [1,2,3] (unchanged) `                                                                                                           |
| `np.flatten(a)`                                                            | Return a flattened **copy**.                                             | `python a = np.array([[1,2],[3,4]]) a.flatten()   # → [1,2,3,4] `                                                                                                                             |
| `np.tile(a, reps)`                                                         | Repeat the whole array block.                                            | `python a = np.array([1,2]) np.tile(a, 3)        # → [1,2,1,2,1,2] np.tile(a,(2,2))     # → [[1,2,1,2],[1,2,1,2]] `                                                                           |
| `np.repeat(a, repeats, axis=None)`                                         | Repeat elements of an array.                                             | `python a = np.array([1,2,3]) np.repeat(a, 2)       # → [1,1,2,2,3,3] b = np.array([[1,2],[3,4]]) np.repeat(b,2,axis=0) # → [[1,2],[1,2],[3,4],[3,4]] `                                       |
| `np.sort(a, axis=-1)`                                                      | Sort elements along axis.                                                | `python a = np.array([[3,1],[2,4]]) np.sort(a, axis=1) # → [[1,3],[2,4]] `                                                                                                                    |
| `np.argsort(a, axis=-1)`                                                   | Indices that would sort array.                                           | `python a = np.array([40,10,30]) np.argsort(a) # → [1,2,0] `                                                                                                                                  |
| `np.unique(a, return_index=False, return_counts=False)`                    | Unique elements, optionally with indices and counts.                     | `python a = np.array([1,2,2,3,3,3]) np.unique(a) # → [1,2,3] np.unique(a, return_counts=True) # → (array([1,2,3]), array([1,2,3])) `                                                          |
| `np.where(condition, x, y)`                                                | Pick from `x` or `y` based on condition. Or indices if only condition.   | `python a = np.array([1,2,3,4,5]) np.where(a%2==0, 'even','odd') # → ['odd','even','odd','even','odd'] np.where(a>3) # → (array([3,4]),) indices `                                            |
| Copy vs View: `.copy()`, `.view()` methods.                                | `.copy()` makes independent data, `.view()` shares same data buffer.     | `python a = np.array([1,2,3]) b = a.view(); c = a.copy() b[0] = 99 a  # → [99,2,3] (changed too!) c[0] = 100 a  # still [99,2,3] (independent) `                                              |

---

## Examples using some of the ones you listed + combining

Let’s do a more involved example showing many of these together:

```python
import numpy as np

# Create a 1D array of 6 values
a = np.arange(1, 7)        # [1,2,3,4,5,6]

# Reshape into 2×3
b = a.reshape((2,3))       # [[1,2,3],[4,5,6]]

# Repeat rows
c = np.repeat(b, 2, axis=0) # repeat each row, now shape is 4×3
# Tile columns
d = np.tile(b, (1,2))       # tile b across columns, now shape 2×6

# Transpose
bt = b.T                   # 3×2

# Do some math
e = np.multiply(b, 10)     # scale
f = np.exp(b)              # e to power of each element
g = np.sqrt(b)             # sqrt

# Linear algebra
I = np.eye(3)
invI = np.linalg.inv(I)    # still I

# Sum, dot, matmul
s0 = np.sum(b, axis=0)      # sum along columns → shape (3,)
s1 = np.sum(b, axis=1)      # along rows → shape (2,)
dot_prod = np.dot(b, bt)    # matrix × matrix → 2×2
mat = np.matmul(b, bt)      # same here

# Log, round
h = np.log(b)               # natural log
r = np.round(h, 3)          # rounded to 3 decimals

# Print nicely
np.set_printoptions(precision=3, suppress=True)
print("b:", b)
print("h:", h)
print("r:", r)
```

---

## More Functions (just names with short notes)

* **Cumulative ops**

  * `np.cumsum` / `np.cumprod` – cumulative sum / product along axes.
  * `np.cummax`, `np.cummin` – cumulative max/min.

* **Differences & gradients**

  * `np.diff` – differences between successive elements.
  * `np.ediff1d` – flattened differences.
  * `np.gradient` – numerical derivative / slope estimates.

* **Signal processing**

  * `np.convolve` – 1D convolution.
  * `np.correlate` – cross-correlation.
  * `np.fft.fft` / `np.fft.ifft` – Fourier transform / inverse.

* **Polynomials**

  * `np.poly` / `np.poly1d` – polynomial utilities (roots, evaluation).

* **Grid / sequences**

  * `np.linspace`, `np.logspace` – evenly spaced values (linear/log scale).
  * `np.meshgrid` – make coordinate grids for functions/plots.

* **Indexing helpers**

  * `np.where`, `np.nonzero` – conditional indices / positions of nonzero.

* **Sorting / ranking**

  * `np.sort`, `np.argsort`, `np.lexsort` – sorting routines.
  * `np.partition` / `np.argpartition` – partial sort (kth element).
  * `np.unique` – unique values + counts/indices.

* **Histograms & counts**

  * `np.histogram` – histogram bin counts.
  * `np.bincount` – frequency counts of non-negative ints.

* **Rounding / NaN utilities**

  * `np.ceil`, `np.floor`, `np.trunc`, `np.modf` – rounding variants.
  * `np.isnan`, `np.isfinite`, `np.nan_to_num` – NaN handling.

* **Interpolation & log tricks**

  * `np.interp` – 1D linear interpolation.
  * `np.logaddexp`, `np.logaddexp2` – stable log-sum-exp.

* **Bitwise / Boolean**

  * `np.bitwise_and`, `np.bitwise_or`, `np.bitwise_xor`, `np.invert`.
  * `np.logical_and`, `np.logical_or`, `np.logical_not`.

* **Set operations**

  * `np.union1d`, `np.intersect1d`, `np.setdiff1d`, `np.setxor1d`.

* **Random sampling**

  * `np.random` – submodule with `rand`, `randn`, `choice`, `permutation`, `normal`, etc.

* **Searching**

  * `np.searchsorted` – binary search in sorted array.

---

# 🔹 Extra NumPy Functions with Examples

| Function              | Description                             | Example                                                                      |
| --------------------- | --------------------------------------- | ---------------------------------------------------------------------------- |
| **`np.cumsum`**       | Cumulative sum along an axis.           | `np.cumsum([1,2,3,4]) → [1,3,6,10]`                                          |
| **`np.cumprod`**      | Cumulative product.                     | `np.cumprod([1,2,3,4]) → [1,2,6,24]`                                         |
| **`np.cummax`**       | Running max.                            | `np.cummax([3,1,4,2]) → [3,3,4,4]`                                           |
| **`np.cummin`**       | Running min.                            | `np.cummin([3,1,4,2]) → [3,1,1,1]`                                           |
| **`np.diff`**         | Successive differences.                 | `np.diff([1,2,4,7]) → [1,2,3]`                                               |
| **`np.ediff1d`**      | Flattened differences.                  | `np.ediff1d([10,15,20]) → [5,5]`                                             |
| **`np.gradient`**     | Numerical gradient (derivative approx). | `np.gradient([1,2,4,7,11]) → [1,1.5,2.5,3.5,4]`                              |
| **`np.convolve`**     | Convolution.                            | `np.convolve([1,2],[3,4]) → [3,10,8]`                                        |
| **`np.correlate`**    | Cross-correlation.                      | `np.correlate([1,2,3],[0,1,0.5]) → [2.5,4,3]`                                |
| **`np.fft.fft`**      | Fast Fourier Transform.                 | `np.fft.fft([0,1,0,0]) → [1.+0.j, 0.-1.j, -1.+0.j, 0.+1.j]`                  |
| **`np.fft.ifft`**     | Inverse FFT.                            | `np.fft.ifft([1,0,-1,0]) → [0.+0.j, 0.5+0.j, 0.+0.j, 0.5+0.j]`               |
| **`np.poly`**         | Polynomial coefficients from roots.     | `np.poly([1,2]) → [1, -3, 2]`                                                |
| **`np.poly1d`**       | Polynomial object.                      | `p=np.poly1d([1,-3,2]); p(5) → 12`                                           |
| **`np.linspace`**     | Evenly spaced numbers.                  | `np.linspace(0,1,5) → [0,0.25,0.5,0.75,1]`                                   |
| **`np.logspace`**     | Logarithmically spaced.                 | `np.logspace(1,3,3) → [10.,100.,1000.]`                                      |
| **`np.meshgrid`**     | Coordinate grid.                        | `X,Y=np.meshgrid([1,2,3],[4,5]); X→[[1,2,3],[1,2,3]]`                        |
| **`np.where`**        | Conditional selection.                  | `np.where([True,False,True],[1,2,3]) → [1,3]`                                |
| **`np.nonzero`**      | Indices of non-zeros.                   | `np.nonzero([0,2,0,3]) → (array([1,3]),)`                                    |
| **`np.sort`**         | Sort values.                            | `np.sort([3,1,2]) → [1,2,3]`                                                 |
| **`np.argsort`**      | Indices for sorting.                    | `np.argsort([3,1,2]) → [1,2,0]`                                              |
| **`np.lexsort`**      | Sort by multiple keys.                  | `np.lexsort((b,a))` sorts by `a` then `b`.                                   |
| **`np.partition`**    | Partial sort (kth element in place).    | `np.partition([3,1,2,5],2) → [2,1,3,5]`                                      |
| **`np.argpartition`** | Indices for partial sort.               | `np.argpartition([3,1,2,5],2) → [1,2,0,3]`                                   |
| **`np.unique`**       | Unique elements sorted.                 | `np.unique([1,2,2,3]) → [1,2,3]`                                             |
| **`np.histogram`**    | Histogram bin counts.                   | `np.histogram([1,2,1], bins=[0,1,2,3]) → (array([0,2,1]), array([0,1,2,3]))` |
| **`np.bincount`**     | Count occurrences of integers.          | `np.bincount([0,1,1,2]) → [1,2,1]`                                           |
| **`np.ceil`**         | Round up.                               | `np.ceil([1.2,3.7]) → [2.,4.]`                                               |
| **`np.floor`**        | Round down.                             | `np.floor([1.2,3.7]) → [1.,3.]`                                              |
| **`np.trunc`**        | Truncate decimal.                       | `np.trunc([-1.7,1.7]) → [-1.,1.]`                                            |
| **`np.modf`**         | Split frac & int parts.                 | `np.modf([1.5,-2.3]) → ([0.5,-0.3],[1.,-2.])`                                |
| **`np.isnan`**        | Check NaN.                              | `np.isnan([1,np.nan]) → [False,True]`                                        |
| **`np.nan_to_num`**   | Replace NaN with 0.                     | `np.nan_to_num([np.nan,1]) → [0.,1.]`                                        |
| **`np.interp`**       | Linear interpolation.                   | `np.interp(2.5,[1,2,3],[10,20,30]) → 25`                                     |
| **`np.logaddexp`**    | Stable `log(exp(x)+exp(y))`.            | `np.logaddexp(1,2) ≈ 2.313`                                                  |
| **`np.logaddexp2`**   | Base-2 version.                         | `np.logaddexp2(2,3) ≈ 3.322`                                                 |
| **`np.bitwise_and`**  | Bitwise AND.                            | `np.bitwise_and(6,3) → 2`                                                    |
| **`np.bitwise_or`**   | Bitwise OR.                             | `np.bitwise_or(6,3) → 7`                                                     |
| **`np.logical_and`**  | Elementwise AND (boolean).              | `np.logical_and([True,False],[True,True]) → [True,False]`                    |
| **`np.union1d`**      | Union of sets.                          | `np.union1d([1,2],[2,3]) → [1,2,3]`                                          |
| **`np.intersect1d`**  | Intersection.                           | `np.intersect1d([1,2],[2,3]) → [2]`                                          |
| **`np.setdiff1d`**    | Set difference.                         | `np.setdiff1d([1,2,3],[2]) → [1,3]`                                          |
| **`np.setxor1d`**     | Symmetric difference.                   | `np.setxor1d([1,2],[2,3]) → [1,3]`                                           |
| **`np.random`**       | Random numbers (module).                | `np.random.rand(2,2)` → random 2×2 floats                                    |
| **`np.searchsorted`** | Index to insert while keeping order.    | `np.searchsorted([1,3,5],4) → 2`                                             |

---

# 🔹 Tricky NumPy Questions

---

## 1. Difference: **`np.multiply` vs `np.dot` vs `np.matmul`**

| Function              | Meaning                                                                                                     | Example                                     | Output              |
| --------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------- |
| **`np.multiply`**     | Element-wise multiplication (Hadamard product).                                                             | `np.multiply([[1,2],[3,4]], [[5,6],[7,8]])` | `[[ 5 12] [21 32]]` |
| **`np.dot`**          | Matrix multiplication (2D) OR inner product (1D).                                                           | `np.dot([[1,2],[3,4]], [[5,6],[7,8]])`      | `[[19 22] [43 50]]` |
| **`np.matmul` (`@`)** | Strict matrix multiplication (no elementwise). Same as `dot` for 2D but works differently with higher dims. | `np.matmul([[1,2],[3,4]], [[5,6],[7,8]])`   | `[[19 22] [43 50]]` |

👉 **Key Trick:**

* `multiply` → elementwise
* `dot` → scalar inner product (1D) OR matrix product (2D)
* `matmul` → always matrix multiplication, even for >2D

---

## 2. Difference: **`np.array()` vs `np.asarray()`**

* `np.array()` → Always makes a copy by default.
* `np.asarray()` → Returns the same object if already an ndarray (no copy).

```python
a = np.arange(5)
b = np.array(a)     # makes a new copy
c = np.asarray(a)   # just a view (no copy)
a[0] = 99
print(b[0], c[0])   # b unaffected, c affected
```

**Output:** `0 99`

👉 **Interview Tip:** Use `asarray` if you want to avoid unnecessary copies (faster).

---

## 3. Difference: **`np.reshape` vs `np.resize`**

* `reshape` → Just changes *view* of same data (must fit size).
* `resize` → Changes shape, repeating/truncating data if needed.

```python
a = np.array([1,2,3,4])
print(np.reshape(a, (2,2)))
print(np.resize(a, (2,3)))
```

**Output:**

```
[[1 2]
 [3 4]]          # reshape fits exactly
[[1 2 3]
 [4 1 2]]        # resize repeats
```

---

## 4. Difference: **`np.ravel` vs `np.flatten`**

* `ravel` → Returns *view* if possible (no extra memory).
* `flatten` → Always returns a *copy*.

```python
a = np.array([[1,2],[3,4]])
r = a.ravel()
f = a.flatten()
a[0,0] = 99
print(r[0], f[0])
```

**Output:** `99 1`

👉 **Interview Trick:** `ravel` is memory-efficient, `flatten` is safer if you want an independent copy.

---

## 5. Difference: **`np.all()` vs `np.any()`**

* `all` → Checks if *all* values are `True`.
* `any` → Checks if *at least one* value is `True`.

```python
a = np.array([0,1,2])
print(np.all(a))   # False (because of 0)
print(np.any(a))   # True  (because of 1,2)
```

---

## 6. Difference: **`np.argmax` vs `np.argsort`**

* `argmax` → Index of max value.
* `argsort` → Indices that would sort array.

```python
a = np.array([10,30,20])
print(np.argmax(a))   # 1
print(np.argsort(a))  # [0 2 1]
```

---

## 7. Difference: **`np.copy` vs `np.view`**

* `copy` → Deep copy (independent).
* `view` → Shallow copy (shares data).

```python
a = np.array([1,2,3])
b = a.copy()
c = a.view()
a[0] = 99
print(b[0], c[0])
```

**Output:** `1 99`

---

## 8. Difference: **`np.vstack`, `np.hstack`, `np.stack`**

```python
a = np.array([1,2])
b = np.array([3,4])

print(np.vstack((a,b)))  # vertical → [[1 2],[3 4]]
print(np.hstack((a,b)))  # horizontal → [1 2 3 4]
print(np.stack((a,b), axis=0))  # new axis → [[1 2],[3 4]]
```

👉 **Trick:** `stack` lets you control axis explicitly.

---

## 9. Difference: **`np.isnan` vs `np.isinf` vs `np.isfinite`**

```python
a = np.array([1, np.nan, np.inf, -np.inf])
print(np.isnan(a))    # [False True False False]
print(np.isinf(a))    # [False False True True]
print(np.isfinite(a)) # [True False False False]
```

---

## 10. Difference: **`np.random.rand` vs `np.random.randn`**

* `rand` → Uniform distribution (0,1).
* `randn` → Normal distribution (mean=0, std=1).

```python
np.random.rand(2)   # [0.56, 0.12]
np.random.randn(2)  # [1.34, -0.23]
```

---