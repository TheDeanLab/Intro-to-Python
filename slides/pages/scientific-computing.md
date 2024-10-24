---
layout: center
---

# Scientific Computing Basics

---

# Introducing ```NumPy``` (<b>Num</b>erical <b>Py</b>thon)
Effecient mathematical operations on N-dimensional data.

First, lets import the library:
```python
import numpy as np
```

- ```Lists``` are versatile in that they can store heterogeneous data and can have "ragged" dimensions.

- Many applications, however, involve operations on rectangular N x M (or more) arrays of the same type, e.g. <b>matrix operations</b>. This is where NumPy really shines!

- If you have used MATLAB, NumPy will be very familiar!
<br>
#### Fundemental type: <b>numpy.ndarray</b>
Create using any rectangular list (or tuple) of a single type:
```python
arr = np.array([[1,2,3,4],[5,6,7,8]])
```

- Can also "initialize" with ```np.zeros, np.ones, np.empty, np.random```, and more.

---

# Introducing ```NumPy``` (<b>Num</b>erical <b>Py</b>thon)

- The power of ```ndarray``` is the ability to perform mathematical operations on them:

<table><tbody><tr>
<td>

 ```python
# scalar operations...
a = np.array([[1,2,3,4],[5,6,7,8]])
a = a * 2 + 1
print("a:", a)
# matrix operations...
b = np.ones(a.shape)
print("b:", b)
print("a + b:", a + b)
 ```

</td>
<td>

 ```console
a: [[ 3  5  7  9]
 [11 13 15 17]]
b: [[1. 1. 1. 1.]
 [1. 1. 1. 1.]]
a + b: [[ 4.  6.  8. 10.]
 [12. 14. 16. 18.]]
 ```

</td>
</tr></tbody></table>

- In addition to <i>slicing</i> (recall w/ ```list```), ```ndarray``` supports <b>Integer</b> and <b>Boolean</b> indexing:

<table><tbody><tr>
<td>

 ```python
# create some random integers
a = np.random.randint(-10,10, size=(3,3))
print(a)
# access only diagonals (Integer indexing)
print("diag:", a[[0,1,2],[0,1,2]])
# acess only > 0 (Boolean indexing)
print("positive:", a[a > 0])
 ```

</td>
<td>

 ```console
[[-5 -6 -9]
 [-7  6  7]
 [ 9  0  2]]
diag: [-5  6  2]
positive: [6 7 9 2]
 ```

</td>
</tr></tbody></table>

---

# Broadcasting 

NumPy handles operations between differently sized arrays by <i>stretching</i> along the larger axis.

<table><tbody>

<tr>
<td>
<img src="https://numpy.org/doc/stable/_images/broadcasting_1.png" width="288" height="128">
</td>
<td>

 ```python
 a = np.array([1,2,3])
 print(a * 2)
 ```
 ```console
 [2, 4, 6]
 ```

</td>
</tr>

<tr>
<td>
<img src="https://numpy.org/doc/stable/_images/broadcasting_2.png" width="288" height="128">
</td>
<td>

 ```python
 a = np.array([[0, 0, 0],[10,10,10],[20,20,20],[30,30,30]])
 b = np.array([1,2,3])
 print(a + b)
 ```
 ```console
 [[ 1  2  3], [11 12 13], [21 22 23], [31 32 33]]
 ```

</td>
</tr>

<tr>
<td>
<img src="https://numpy.org/doc/stable/_images/broadcasting_4.png" width="288" height="128">
</td>
<td>

 ```python
 a = np.array([0, 10, 20, 30])
 b = np.array([1, 2, 3])
 print( a[:, np.newaxis] + b )
 ```
 ```console
 [[ 1  2  3], [11 12 13], [21 22 23], [31 32 33]]
 ```

</td>
</tr>

</tbody></table>

---

# And so much more!

** **
<br>

### Basic Math: ```np.sin, np.cos, np.exp, np.log, ...```

### Statistics: ```np.mean, np.std, np.max, np.sum, ...```

### Linear algebra: ```np.transpose, np.invert, np.dot, np.cross, ...```

### Fourier domain: ```np.fft.fft/fft2, np.fft.ifft/ifft2, ...```
<br><br>
## Let's not get bogged down and instead see what ```NumPy``` can do!
## But as usual, <a href="https://numpy.org/">read the docs</a>. 

---

# Working with Images