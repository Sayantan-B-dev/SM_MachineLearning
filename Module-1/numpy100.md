# Tricky NumPy Interview Q\&A 

### **Q1. Difference between `*`, `.dot()`, and `@`?**

**Answer:**

* `*` → elementwise multiplication
* `.dot()` → matrix multiplication (or dot product)
* `@` → same as `.dot()` (PEP 465 operator)

```python
import numpy as np
a = np.array([[1,2],[3,4]])
b = np.array([[2,0],[1,3]])

print(a * b)      # Elementwise
print(a.dot(b))   # Matrix multiplication
print(a @ b)      # Same as above
```

📌 **Explanation:** Interviewers check if you confuse elementwise with linear algebra.

---

### **Q2. Difference between `np.array([1,2,3,4])` and `np.arange(1,5)`?**

```python
np.array([1,2,3,4])   # Explicit list → [1 2 3 4]
np.arange(1,5)        # Range-like → [1 2 3 4]
```

📌 **Explanation:** Both produce same output here, but `np.arange` generates numbers dynamically (good for ranges).

---

### **Q3. Why is `np.array([0.1,0.2,0.3]) == 0.3` not always True?**

```python
arr = np.array([0.1,0.2,0.3])
print(arr == 0.3)   # [False False  True]
```

📌 **Explanation:** Floating point precision: `0.1+0.2` ≠ `0.3` exactly.

---

### **Q4. Difference between `np.copy()` and `np.view()`?**

```python
a = np.array([1,2,3])
b = a.view()
c = a.copy()
a[0] = 99
print(b)  # Changes → [99  2  3]
print(c)  # Stays   → [1 2 3]
```

📌 **Explanation:** `view` shares memory, `copy` makes independent data.

---

### **Q5. Explain `reshape(-1,1)` vs `reshape(1,-1)`**

```python
arr = np.array([1,2,3,4])
print(arr.reshape(-1,1))  # Column vector
print(arr.reshape(1,-1))  # Row vector
```

📌 **Explanation:** `-1` auto-computes size; first form → vertical, second → horizontal.

---

### **Q6. Difference between `np.linspace` and `np.arange`?**

```python
print(np.linspace(0,1,5))   # Equal spacing → [0. 0.25 0.5 0.75 1.]
print(np.arange(0,1,0.25))  # Step-based    → [0. 0.25 0.5 0.75]
```

📌 **Explanation:** `linspace` includes endpoint, `arange` doesn’t always.

---

### **Q7. Why is `np.sum(arr)` slower than `sum()` for small arrays but faster for large ones?**

📌 **Explanation:** Python’s `sum` has less overhead for tiny arrays, but NumPy is C-optimized and shines on big arrays.

---

### **Q8. Broadcasting in `arr + 5` vs `arr + arr.T`**

```python
arr = np.arange(4).reshape(2,2)
print(arr + 5)      # Scalar → adds to each
print(arr + arr.T)  # Needs shape alignment
```

📌 **Explanation:** Scalar broadcasting is simple; with arrays, NumPy aligns shapes.

---

### **Q9. What happens with `arr.flags.writeable = False`?**

```python
arr = np.array([1,2,3])
arr.flags.writeable = False
# arr[0] = 10  # Error: assignment destination is read-only
```

📌 **Explanation:** Locks array → read-only.

---

### **Q10. Difference between `ravel()` and `flatten()`?**

```python
a = np.array([[1,2],[3,4]])
print(a.ravel())   # May return view
print(a.flatten()) # Always copy
```

📌 **Explanation:** `ravel` is memory-efficient, `flatten` guarantees independent copy.

---

### **Q11. Difference between `arr.T` and `arr.transpose((0,1))`?**

```python
a = np.arange(6).reshape(2,3)
print(a.T)                  # Transpose
print(a.transpose((0,1)))   # No change
```

📌 **Explanation:** `.T` reverses axes, while `transpose` lets you specify order.

---

### **Q12. Difference between `vstack` and `hstack`?**

```python
a = np.array([1,2])
b = np.array([3,4])
print(np.vstack([a,b]))  # [[1 2] [3 4]]
print(np.hstack([a,b]))  # [1 2 3 4]
```

📌 **Explanation:** Vertical stack = rows, Horizontal = columns.

---

### **Q13. Why does `np.where(condition, x, y)` return arrays?**

```python
arr = np.array([1,2,3])
print(np.where(arr>2, 10, 20))  # [20 20 10]
```

📌 **Explanation:** Works elementwise; even if condition is single bool, result is array.

---

### **Q14. Difference between `isnan()` and `isfinite()`?**

```python
arr = np.array([1, np.nan, np.inf])
print(np.isnan(arr))   # [False  True False]
print(np.isfinite(arr))# [ True False False]
```

📌 **Explanation:** `isnan` checks only NaN, `isfinite` excludes both NaN & Inf.

---

### **Q15. How to check if two arrays share memory?**

```python
a = np.arange(10)
b = a[::2]
print(np.shares_memory(a,b))  # True
```

📌 **Explanation:** Important in debugging performance issues.

---

### **Q16. Why `np.array([1,2,3]) + np.array([4,5])` errors but broadcasting works in 2D?**

```python
# Error:
# np.array([1,2,3]) + np.array([4,5])
a = np.array([[1],[2],[3]])
b = np.array([4,5])
print(a+b)  # Works → broadcasting
```

📌 **Explanation:** Shapes (3,) and (2,) don’t align; (3,1) and (2,) do.

---

### **Q17. What does `np.zeros((2,3), dtype=bool)` create?**

```python
print(np.zeros((2,3), dtype=bool))
# [[False False False]
#  [False False False]]
```

📌 **Explanation:** False values, not numeric zeros.

---

### **Q18. Difference between `concatenate` and `stack`?**

```python
a = np.array([1,2])
b = np.array([3,4])
print(np.concatenate([a,b]))  # [1 2 3 4]
print(np.stack([a,b]))        # [[1 2] [3 4]]
```

📌 **Explanation:** `stack` adds new axis, `concatenate` joins existing.

---

### **Q19. Why does `arr == np.nan` always return False?**

```python
arr = np.array([np.nan])
print(arr == np.nan)   # [False]
print(np.isnan(arr))   # [True]
```

📌 **Explanation:** NaN is never equal to itself; must use `isnan`.

---

### **Q20. What’s difference between `astype()` and `view()` for dtype conversion?**

```python
a = np.array([1,2,3], dtype=np.int32)
print(a.astype(np.float32))  # Converts values
print(a.view(np.float32))    # Reinterprets bits
```

📌 **Explanation:** `astype` changes values safely, `view` only reinterprets memory.

---

### **Q21. What is the difference between `arr.tolist()` and `list(arr)`?**

```python
a = np.array([[1,2],[3,4]])
print(a.tolist())  # [[1,2],[3,4]]
print(list(a))     # [array([1,2]), array([3,4])]
```

📌 **Explanation:** `tolist()` recursively converts to nested Python lists, while `list()` only converts the outermost array.

---

### **Q22. Why does slicing not copy data in NumPy?**

```python
a = np.arange(5)
b = a[1:4]
b[0] = 99
print(a)  # [ 0 99  2  3  4]
```

📌 **Explanation:** NumPy slicing returns a **view** for efficiency → shares memory.

---

### **Q23. Difference between `np.allclose()` and `np.array_equal()`?**

```python
a = np.array([0.1+0.2, 0.3])
b = np.array([0.3, 0.3])
print(np.array_equal(a,b))  # False
print(np.allclose(a,b))     # True
```

📌 **Explanation:** `array_equal` needs exact equality, `allclose` allows floating tolerance.

---

### **Q24. Why is `(a == b).all()` not same as `a is b`?**

```python
a = np.array([1,2,3])
b = np.array([1,2,3])
print((a == b).all())  # True (values equal)
print(a is b)          # False (different objects)
```

📌 **Explanation:** Checks values vs identity.

---

### **Q25. Why does reshaping with `-1` work?**

```python
a = np.arange(12)
print(a.reshape(3, -1))  # (3,4)
```

📌 **Explanation:** NumPy auto-calculates dimension to fit total size.

---

### **Q26. Difference between `flatten(order='C')` and `flatten(order='F')`?**

```python
a = np.array([[1,2],[3,4]])
print(a.flatten('C'))  # [1 2 3 4] (row-major)
print(a.flatten('F'))  # [1 3 2 4] (column-major)
```

📌 **Explanation:** `'C'` → row-first, `'F'` → column-first (Fortran style).

---

### **Q27. Why does `np.argsort()` differ from `np.sort()`?**

```python
a = np.array([40,10,30])
print(np.sort(a))     # [10 30 40]
print(np.argsort(a))  # [1 2 0]
```

📌 **Explanation:** `argsort` gives indices that would sort, not sorted values.

---

### **Q28. Difference between `arr.max(axis=0)` and `arr.max(axis=1)`?**

```python
a = np.array([[1,5,2],[7,0,3]])
print(a.max(axis=0))  # [7 5 3] (column-wise)
print(a.max(axis=1))  # [5 7]   (row-wise)
```

📌 **Explanation:** Axis 0 → down columns, Axis 1 → across rows.

---

### **Q29. Why is `np.random.seed()` used?**

```python
np.random.seed(42)
print(np.random.rand(3))  # Same output each run
```

📌 **Explanation:** Ensures reproducibility of random results.

---

### **Q30. Why does `arr.astype(int)` sometimes truncate floats?**

```python
a = np.array([3.9, -2.7])
print(a.astype(int))  # [ 3 -2]
```

📌 **Explanation:** Converts by truncating (toward zero), not rounding.

---

### **Q31. What’s the difference between `np.floor()`, `np.ceil()`, and `np.trunc()`?**

```python
a = np.array([3.7, -3.7])
print(np.floor(a))  # [ 3. -4.]
print(np.ceil(a))   # [ 4. -3.]
print(np.trunc(a))  # [ 3. -3.]
```

📌 **Explanation:** Floor → down, Ceil → up, Trunc → toward zero.

---

### **Q32. Why does `np.mean([])` return `nan`?**

```python
print(np.mean([]))  # nan + RuntimeWarning
```

📌 **Explanation:** Division by zero length array → `nan`.

---

### **Q33. Difference between `np.unique()` and `set()`?**

```python
a = np.array([3,1,2,1,3])
print(np.unique(a))   # [1 2 3] sorted
print(set(a))         # {1,2,3} unordered
```

📌 **Explanation:** `np.unique` always returns sorted array.

---

### **Q34. What happens if shapes don’t align for broadcasting?**

```python
a = np.ones((3,2))
b = np.ones((3,))
# print(a + b)  # ValueError
```

📌 **Explanation:** Broadcasting requires dimensions to match or be 1.

---

### **Q35. Why does `np.dot([1,2],[3,4])` work without arrays?**

```python
print(np.dot([1,2],[3,4]))  # 11
```

📌 **Explanation:** NumPy auto-converts lists to arrays.

---

### **Q36. What’s the difference between `np.cumsum()` and `np.sum()`?**

```python
a = np.array([1,2,3])
print(np.sum(a))     # 6
print(np.cumsum(a))  # [1 3 6]
```

📌 **Explanation:** `cumsum` keeps running total, `sum` is final total.

---

### **Q37. Why does `np.any([])` return `False` and `np.all([])` return `True`?**

```python
print(np.any([]))  # False
print(np.all([]))  # True
```

📌 **Explanation:** Logical identity → empty OR = False, empty AND = True.

---

### **Q38. Difference between `np.dot()` and `np.outer()`?**

```python
a = np.array([1,2])
b = np.array([3,4])
print(np.dot(a,b))    # 11 (scalar)
print(np.outer(a,b))  # [[3 4] [6 8]]
```

📌 **Explanation:** Dot → scalar inner product; Outer → full matrix.

---

### **Q39. Why is `np.log(0)` not valid?**

```python
print(np.log(0))  # -inf + RuntimeWarning
```

📌 **Explanation:** Logarithm of 0 is undefined → returns `-inf`.

---

### **Q40. What’s difference between `arr.item()` and `arr.tolist()` for scalars?**

```python
a = np.array(5)
print(a.item())    # 5 (Python scalar)
print(a.tolist())  # 5 (Python scalar too)
```

📌 **Explanation:** Both give Python scalars, but `item()` is faster, specialized for single elements.

---

### **Q41. Difference between `arr.copy()` and `arr[:]`?**

```python
a = np.array([1,2,3])
b = a[:]      # View (shares memory)
c = a.copy()  # Independent copy
a[0] = 99
print(b)  # [99  2  3]
print(c)  # [1 2 3]
```

📌 **Explanation:** `[:]` is still a view; only `copy()` is independent.

---

### **Q42. Why does `np.array([True, False]) + np.array([1,2])` work?**

```python
print(np.array([True, False]) + np.array([1,2]))  # [2 2]
```

📌 **Explanation:** Boolean `True=1`, `False=0`. NumPy auto-casts.

---

### **Q43. Difference between `np.ceil_divide` (not in NumPy) vs `np.floor_divide`?**

```python
a = np.array([7])
print(a // 3)                  # 2 (floor divide)
print(np.floor_divide(a, 3))   # 2
```

📌 **Explanation:** NumPy only has **floor division**, not ceil.

---

### **Q44. What’s `np.may_share_memory()` vs `np.shares_memory()`?**

```python
a = np.arange(10)
b = a[::2]
print(np.shares_memory(a,b))     # True
print(np.may_share_memory(a,b))  # True (conservative check)
```

📌 **Explanation:** `may_share_memory` may return True even if not, but never misses.

---

### **Q45. Why does `np.dot(a,a)` for 1D differ from 2D?**

```python
a = np.array([1,2])
print(np.dot(a,a))  # 5 (scalar)

b = np.array([[1,2]])
print(np.dot(b,b.T)) # [[5]]
```

📌 **Explanation:** 1D arrays → scalar; 2D arrays → matrix.

---

### **Q46. Why is `np.matmul` preferred over `np.dot`?**

```python
a = np.ones((3,1,2))
b = np.ones((3,2,4))
print(np.matmul(a,b).shape)  # (3,1,4)
```

📌 **Explanation:** `matmul` supports batch matrix multiplication, `dot` doesn’t.

---

### **Q47. What is difference between `arr.flat` and `arr.flatten()`?**

```python
a = np.array([[1,2],[3,4]])
print(list(a.flat))     # Iterator → [1,2,3,4]
print(a.flatten())      # Array copy → [1 2 3 4]
```

📌 **Explanation:** `flat` is an **iterator**, `flatten` returns new array.

---

### **Q48. Why does `np.clip()` exist?**

```python
a = np.array([1,5,10])
print(np.clip(a, 2, 7))  # [2 5 7]
```

📌 **Explanation:** Clips values into \[min,max] range (useful in ML, normalization).

---

### **Q49. Why does `arr[None, :]` add a dimension?**

```python
a = np.array([1,2,3])
print(a.shape)         # (3,)
print(a[None,:].shape) # (1,3)
```

📌 **Explanation:** `None` is alias for `np.newaxis`, adds axis.

---

### **Q50. Difference between `np.tile()` and `np.repeat()`?**

```python
a = np.array([1,2])
print(np.tile(a, 3))    # [1 2 1 2 1 2]
print(np.repeat(a, 3))  # [1 1 1 2 2 2]
```

📌 **Explanation:** `tile` repeats whole array, `repeat` repeats each element.

---

### **Q51. What’s the difference between `np.all()` and `np.logical_and.reduce()`?**

```python
a = np.array([True, True, False])
print(np.all(a))                # False
print(np.logical_and.reduce(a)) # False
```

📌 **Explanation:** Same result; but `logical_and.reduce` is more explicit and flexible with axes.

---

### **Q52. Why does `np.prod([])` return 1?**

```python
print(np.prod([]))  # 1
```

📌 **Explanation:** Multiplicative identity = 1 (like empty sum = 0).

---

### **Q53. Difference between `np.nonzero()` and `np.where()`?**

```python
a = np.array([0,2,0,4])
print(np.nonzero(a))        # (array([1,3]),)
print(np.where(a != 0))     # (array([1,3]),)
```

📌 **Explanation:** Both similar, but `where` also supports condition with x,y.

---

### **Q54. Why does `np.argmax([])` throw error?**

```python
# np.argmax([])  # ValueError
```

📌 **Explanation:** No maximum in empty array.

---

### **Q55. What is difference between `np.logical_and(a,b)` and `a & b`?**

```python
a = np.array([True, False])
b = np.array([True, True])
print(np.logical_and(a,b))  # [ True False]
print(a & b)                # [ True False]
```

📌 **Explanation:** Same for booleans, but `&` works bitwise on integers too.

---

### **Q56. Why does `np.corrcoef()` sometimes return NaN?**

```python
a = np.array([1,1,1])
print(np.corrcoef(a))  # nan
```

📌 **Explanation:** Correlation undefined when variance=0.

---

### **Q57. Difference between `np.dot()` and `@` with higher dimensions?**

```python
a = np.ones((2,2,2))
b = np.ones((2,2,2))
print(np.dot(a,b).shape)   # (2,2,2,2)
print((a@b).shape)         # (2,2,2)
```

📌 **Explanation:** `@` follows strict matrix multiplication rules; `dot` generalizes.

---

### **Q58. Why does `arr == None` not work as expected?**

```python
a = np.array([None, 1])
print(a == None)  # [ True False]
```

📌 **Explanation:** Works elementwise but better use `arr is None` for object check.

---

### **Q59. What’s the difference between `arr.ndim`, `arr.shape`, and `arr.size`?**

```python
a = np.arange(12).reshape(3,4)
print(a.ndim)  # 2
print(a.shape) # (3,4)
print(a.size)  # 12
```

📌 **Explanation:** ndim → dimensions, shape → size per axis, size → total elements.

---

### **Q60. Why does `np.empty()` contain garbage values?**

```python
print(np.empty((2,3)))
```

📌 **Explanation:** `empty` allocates memory but doesn’t initialize → random values.

---

### **Q61. Why does `np.nan == np.nan` return False?**

```python
print(np.nan == np.nan)   # False
print(np.isnan(np.nan))   # True
```

📌 **Explanation:** By IEEE standard, NaN ≠ NaN. Must use `isnan()`.

---

### **Q62. What’s difference between `np.isclose()` and `np.allclose()`?**

```python
a = np.array([1.00001, 2.0])
b = np.array([1.0, 2.0])
print(np.isclose(a,b))   # [ True  True]
print(np.allclose(a,b))  # True
```

📌 **Explanation:** `isclose` → elementwise, `allclose` → overall check.

---

### **Q63. Why does `np.random.choice` with `replace=False` fail if size > population?**

```python
# np.random.choice(5, size=10, replace=False) -> ValueError
```

📌 **Explanation:** Sampling without replacement can’t exceed population size.

---

### **Q64. Difference between `np.fromiter()` and `np.array(list(...))`?**

```python
it = (x*x for x in range(5))
print(np.fromiter(it, dtype=int))  # [0 1 4 9 16]
```

📌 **Explanation:** `fromiter` builds directly from iterator, more memory-efficient.

---

### **Q65. Why does `np.linspace(0,1,3,endpoint=False)` exclude 1?**

```python
print(np.linspace(0,1,3,endpoint=False))  # [0.   0.333 0.667]
```

📌 **Explanation:** `endpoint=False` stops before last value.

---

### **Q66. What’s difference between `arr.astype('int32')` and `arr.astype(np.int32, copy=False)`?**

📌 **Explanation:** With `copy=False`, NumPy avoids copying if already correct dtype.

---

### **Q67. Why does `np.arange(0,1,0.1)` sometimes look weird?**

```python
print(np.arange(0,1,0.1))  
# [0.  0.1 0.2 0.3 ... 0.9]
```

📌 **Explanation:** Floating errors may cause last value ≠ expected. Better: `linspace`.

---

### **Q68. Difference between `np.set_printoptions()` and `round()`?**

```python
a = np.array([1.123456])
np.set_printoptions(precision=2)
print(a)   # [1.12]
```

📌 **Explanation:** `set_printoptions` only affects **display**, not actual values.

---

### **Q69. Why does `np.median()` differ from `np.mean()` for skewed data?**

```python
a = np.array([1,2,100])
print(np.mean(a))   # 34.33
print(np.median(a)) # 2
```

📌 **Explanation:** Median robust to outliers, mean sensitive.

---

### **Q70. What’s difference between `np.linalg.inv()` and `np.linalg.pinv()`?**

```python
a = np.array([[1,2],[3,4]])
print(np.linalg.inv(a))   # Exact inverse
print(np.linalg.pinv(a))  # Pseudo-inverse (handles singular)
```

📌 **Explanation:** `pinv` works even when matrix not invertible.

---

### **Q71. Why is `np.all(np.isnan([]))` True?**

```python
print(np.all(np.isnan([])))  # True
```

📌 **Explanation:** Vacuously true (no counterexample exists in empty array).

---

### **Q72. Difference between `np.save` and `np.savetxt`?**

📌 **Explanation:**

* `save` → binary `.npy` (fast, exact)
* `savetxt` → human-readable text (slower, precision loss).

---

### **Q73. Why does `np.add.reduce()` exist when `np.sum()` already does the job?**

```python
a = np.arange(5)
print(np.add.reduce(a))  # 10
```

📌 **Explanation:** `reduce` is a generalized ufunc method (flexible for custom ops).

---

### **Q74. Why does `np.mean(arr, dtype=np.float64)` matter?**

```python
a = np.array([1,2,3], dtype=np.int8)
print(np.mean(a))                    # 2.0 (default float64)
print(np.mean(a, dtype=np.float32))  # 2.0 (explicit)
```

📌 **Explanation:** Prevents overflow for small dtypes.

---

### **Q75. Difference between `np.random.rand()` and `np.random.randn()`?**

```python
print(np.random.rand(3))   # [0.1 0.7 0.3] uniform [0,1)
print(np.random.randn(3))  # [ 0.5 -1.3 0.9] normal dist
```

📌 **Explanation:** `rand` uniform, `randn` normal distribution.

---

### **Q76. Why does `np.append()` feel slow?**

```python
a = np.array([1,2])
print(np.append(a, [3,4]))  # [1 2 3 4]
```

📌 **Explanation:** It always creates new array → costly in loops.

---

### **Q77. What’s difference between `arr.shape = ...` and `arr.reshape(...)`?**

```python
a = np.arange(6)
a.shape = (2,3)     # In-place
print(a)
print(a.reshape(3,2)) # Returns new view
```

📌 **Explanation:** `reshape` gives new array (view/copy), `shape=` modifies original.

---

### **Q78. Why does `np.arange(3).reshape(2,2)` fail?**

```python
# np.arange(3).reshape(2,2) -> ValueError
```

📌 **Explanation:** Cannot reshape size-3 array into size-4 shape.

---

### **Q79. Difference between `np.min([])` and `np.max([])`?**

```python
# np.min([]) -> ValueError (no identity)
# np.max([]) -> ValueError
```

📌 **Explanation:** Unlike `sum/prod`, min/max need at least one element.

---

### **Q80. Why does `np.convolve([1,2],[3,4])` return 3 elements?**

```python
print(np.convolve([1,2],[3,4]))  # [3 10 8]
```

📌 **Explanation:** Convolution length = `n+m-1`.

---

### **Q81. Why does `np.unique()` return sorted values?**

```python
a = np.array([3,1,2,3])
print(np.unique(a))  # [1 2 3]
```

📌 **Explanation:** By default, it sorts. Use `return_index=True` for original order.

---

### **Q82. Difference between `np.argmax()` and `np.argpartition()`?**

```python
a = np.array([10,50,20,30])
print(np.argmax(a))        # 1 (max index)
print(np.argpartition(a,-2)[-2:])  # indices of top-2
```

📌 **Explanation:** `argmax` → single max, `argpartition` → k-th order stats.

---

### **Q83. Why does `np.histogram()` return two arrays?**

```python
data = [1,2,2,3]
counts, bins = np.histogram(data, bins=3)
print(counts)  # [1 2 1]
print(bins)    # [1.  1.67 2.33 3. ]
```

📌 **Explanation:** One array = bin counts, other = bin edges.

---

### **Q84. Difference between `np.tile()` and `np.repeat()`?**

```python
a = np.array([1,2])
print(np.tile(a,2))   # [1 2 1 2]
print(np.repeat(a,2)) # [1 1 2 2]
```

📌 **Explanation:** `tile` → repeats blocks, `repeat` → repeats elements.

---

### **Q85. Why does `np.linspace(0,1,5,retstep=True)` return step?**

```python
vals, step = np.linspace(0,1,5,retstep=True)
print(vals)  # [0.   0.25 0.5  0.75 1. ]
print(step)  # 0.25
```

📌 **Explanation:** Extra return shows spacing between numbers.

---

### **Q86. What’s difference between `np.pad()` and `np.full()` for padding?**

```python
a = np.array([1,2,3])
print(np.pad(a,(2,2)))        # [0 0 1 2 3 0 0]
print(np.full(7,0))           # [0 0 0 0 0 0 0]
```

📌 **Explanation:** `pad` adds around array, `full` creates standalone array.

---

### **Q87. Why does `np.dot(a,b)` differ from `np.vdot(a,b)`?**

```python
a = np.array([1+2j])
b = np.array([3+4j])
print(np.dot(a,b))   # (-5+10j)
print(np.vdot(a,b))  # (11+0j)
```

📌 **Explanation:** `vdot` conjugates the first array.

---

### **Q88. Why does `np.diag()` behave differently for 1D vs 2D input?**

```python
print(np.diag([1,2,3]))
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
```

📌 **Explanation:**

* 1D → makes diagonal matrix.
* 2D → extracts diagonal.

---

### **Q89. What’s difference between `arr.flat` and `arr.flatten()`?**

```python
a = np.array([[1,2],[3,4]])
print(list(a.flat))     # iterator [1,2,3,4]
print(a.flatten())      # array [1 2 3 4]
```

📌 **Explanation:** `flat` → iterator, `flatten` → copy.

---

### **Q90. Why does `np.nanargmax([1, np.nan, 2])` work?**

```python
print(np.nanargmax([1,np.nan,2]))  # 2
```

📌 **Explanation:** Ignores NaNs, unlike `argmax`.

---

### **Q91. What’s difference between `arr.any()` and `arr.all()`?**

```python
a = np.array([0,1,2])
print(a.any())  # True (at least one non-zero)
print(a.all())  # False (not all non-zero)
```

---

### **Q92. Why does `np.take()` exist when indexing works?**

```python
a = np.array([10,20,30])
print(np.take(a,[2,0]))  # [30 10]
```

📌 **Explanation:** `take` works with out-of-bound modes & axis control.

---

### **Q93. Why does `np.random.seed()` matter?**

```python
np.random.seed(42)
print(np.random.rand(3))  # Always same output
```

📌 **Explanation:** Ensures reproducibility.

---

### **Q94. Difference between `np.empty()` and `np.zeros()`?**

```python
print(np.empty(3))  # [?? random garbage ??]
print(np.zeros(3))  # [0. 0. 0.]
```

📌 **Explanation:** `empty` doesn’t initialize, `zeros` does.

---

### **Q95. Why does `np.corrcoef()` return a 2x2 matrix for two arrays?**

```python
x = [1,2,3]; y=[1,2,4]
print(np.corrcoef(x,y))
```

📌 **Explanation:** Correlation matrix of all pairs.

---

### **Q96. Why does `np.argsort()` differ from `sorted()`?**

```python
a = np.array([30,10,20])
print(np.argsort(a))  # [1 2 0]
print(sorted(a))      # [10,20,30]
```

📌 **Explanation:** `argsort` returns indices, not values.

---

### **Q97. What’s difference between `arr.copy()` and `copy.deepcopy()`?**

📌 **Explanation:**

* `arr.copy()` shallow copy of array.
* `deepcopy()` needed only when nested objects inside dtype=object.

---

### **Q98. Why does `np.round(2.5)` sometimes return 2.0?**

```python
print(np.round(2.5))  # 2.0 (banker’s rounding)
```

📌 **Explanation:** NumPy uses round-to-even strategy.

---

### **Q99. Why does `np.clip()` help prevent overflow?**

```python
a = np.array([1,100,200])
print(np.clip(a,0,150))  # [  1 100 150]
```

📌 **Explanation:** Forces values into safe range.

---

### **Q100. Final trick: Why does `np.allclose(0.1+0.2,0.3)` return True?**

```python
print(0.1+0.2 == 0.3)        # False
print(np.allclose(0.1+0.2,0.3))  # True
```

📌 **Explanation:** Floating precision issue; `allclose` tolerates small errors.

---
