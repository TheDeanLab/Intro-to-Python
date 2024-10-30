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
## ```NumPy``` can do a ton! Let's not get too bogged down...
## But as usual, <a href="https://numpy.org/">read the docs</a>. 

---

# Loading Images

- Numerous packages exist which can load images easily. Some common ones are:

<table><tbody>
<tr><td>Matplotlib</td>
<td>

```python
import matplotlib.pyplot as plt
im = plt.imread('img.png')
```

</td></tr>
<tr><td>PIL</td>
<td>

```python
from PIL import Image
im = Image.open('img.png')
```

</td></tr>
<tr><td>OpenCV</td>
<td>

```python
from cv2
im = cv2.imread('img.png')
```

</td></tr>
<tr><td>Scikit-Image</td>
<td>

```python
from skimage import io
io.imread('img.png')
```

</td></tr>
</tbody></table>

--- 

# Working with Images

- Let's load an image using some of the tools that we're already familiar with:

<table><tbody><tr>
<td>

```python
from skimage import io
import matplotlib.pyplot as plt

im = io.imread("..\\datasets\\images\\ihc.png")
plt.imshow(im)
```

</td>
<td>

<img src="../images/ihc.png" width=180>

</td>
</tr></tbody></table>

- What are some of the properties of our image?


<table><tbody><tr>
<td>

```python
print(type(im))
print(im.shape)
```

</td>
<td>

```console
<class 'numpy.ndarray'>
(512, 512, 3)
```

</td>
</tr></tbody></table>

- Many image analysis packages default to ```numpy.ndarray``` as the image datatype, making them convenient to work with.

---

# Working with Images

### RGB Color

- Note the 3rd dimension on our image: these are the <b>red, green, blue 
(<span style="color:red">R</span>
<span style="color:green">G</span>
<span style="color:blue">B</span>)
</b> components of our image.

<center>
    <img src="../images/ihc_rgb.png" width=512>
</center>

- If we ``print(im)`` directly...

<center>
```console
array([[[156, 118,  81],
        [163, 125,  88],
        [156, 116,  81], 
        ..., etc
```
</center>

... we see that each pixel has the form <b>\[r, g, b\]</b>. Combined these values can form any color, but sometimes it may be useful to work with the RGB channels individually.

--- 

# Working with Images

The ```ndarray``` nature of images allows us manipulate them as with any NumPy array.

<table><tbody><tr>
<td>

##### Cropping

```python
plt.imshow(im[64:128, 64:128, :])
```

</td>
<td>

<img src="../images/ihc_crop.png" width=110>

</td>
</tr></tbody></table>

<table><tbody><tr>
<td>

##### Rotate

```python
plt.imshow(im.swapaxes(0,1))
```

</td>
<td>

<img src="../images/ihc_trans.png" width=110>

</td>
</tr></tbody></table>

<table><tbody><tr>
<td>

##### Downsample

```python
plt.imshow(im[::16, ::16, :])
```

</td>
<td>

<img src="../images/ihc_ds.png" width=110>

</td>
</tr></tbody></table>


--- 

# Statistical Analysis with ```SciPy```

### T-Test for Independence

- Let's use what we have learned to generate some toy data and perform a T-Test for independence.

<table><tbody>

<tr>
<td>

 ```python
import numpy as np
# --------- Generate three random samples ---------
#                     sample   effect   variance
#                     size     size
v0 = np.random.normal(size=20, loc=0.0, scale=1.0)
v1 = np.random.normal(size=20, loc=0.1, scale=1.0)
v2 = np.random.normal(size=20, loc=1.0, scale=1.0)
# plot the data
import seaborn as sns
sns.violinplot(data=[v0, v1, v2], inner='point')
 ```

</td>
<td>

<img src="../images/sci-comp_ttest_violin.png" width=256>
Always <u>visualize your data</u>!

</td>
</tr>

<tr>
<td>

 ```python
import scipy.stats as spstats
result_01 = spstats.ttest_ind(v0, v1)
result_02 = spstats.ttest_ind(v0, v2)
print(f"[0<->1]: P={result_01.pvalue:.4f}, t={result_01.statistic:.4f}")
print(f"[0<->2]: P={result_02.pvalue:.4f}, t={result_02.statistic:.4f}")
 ```

</td>
<td>

T-Test Results:
```console
[0<->1]: P=0.2008, t=-1.3019
[0<->2]: P=0.0065, t=-2.8815
```

</td>
</tr>

</tbody></table>

---

# Statistical Analysis with ```SciPy```

### One-way ANOVA

- <i>"Analysis of variance"</i> using the F-Statistic. Tests the null hypothesis that <u>all means are equal</u>.

<table><tbody>

<tr>
<td>

 ```python
from scipy.stats import f_oneway

# perform the one-way anova on our toy data
result = f_oneway(v0, v1, v2) # include all

print(f"ANOVA results:\nP = {result.pvalue:.4f}\nF = {result.statistic:.4f}")
 ```

</td>
<td>

```console
ANOVA results:
P = 0.0104
F = 4.9470
```

</td>
</tr>
</tbody></table>

- What happens to the ANOVA if we drastically reduce the "effect size (```loc```)" of ```v2``` from ```1.0``` to ```0.2```?

<table><tbody>

<tr>
<td>

 ```python
v2 = np.random.normal(size=20, loc=0.2, scale=1.0)

result = f_oneway(v0, v1, v2)

print(f"ANOVA results:\nP = {result.pvalue:.4f}\nF = {result.statistic:.4f}")
 ```

</td>
<td>

```console
ANOVA results:
P = 0.3401
F = 1.0993
```

</td>
</tr>
</tbody></table>

- ```P > 0.05```: There is no significant difference between our datasets.

---

# Introduction to Machine Learning

--- 

# Data Preparation

- Features vs Targets

- Classification vs Regression

- One-hot encoding?

- Train/Test/Validation split!

--- 

# Regression Models

- Linear vs Non-linear

- Overview of common ones:
    - ElasticNet, Lasso, XGBoost

--- 

# Classification Models

- Binary vs Multiclass

- One-hot encoding

- kNN, SVM, Random Forest