---
layout: center
---

# Introduction to the Python Data Analysis Ecosystem
## Don't re-invent the wheel: 
## find examples and read documentation!


---

# Standard Python has limited functionality and un-optimized for efficiency

<v-click>

- ### So Far: Python data and control structures
	- We have covered the variable types in python, and how to store them in data structures
	- We have covered how to manipulate variables through control structures
</v-click>

<v-click>

- ### Today: Using Python in real-world application
	- Basic Python is very limited (We have covered most of it!)
	- Writing code from scratch is suboptimal for real-world applications
		- Low-level compiled languages e.g. C, Fortran more optimized for speed
		- Many functions have already been written and well-maintained (waste of time)
		- Many projects can be solved by mix-and-match
	- **Your objective:** Only write code for data reading and manipulation to use Python libraries and to operate on results
</v-click>


---

# Diverse ecosystem of well-maintained libraries
Different libraries for different uses

- Libraries are 'plugins' providing a set of functions for particular purposes that extend the basic Python. 
- Find and install through `conda` or `pip`. 
- For custom libraries, installation instructions will be provided in the GitHub repository usually  
<div class="flex flex-col items-center">
  <div>
    <img src="/images/python-stack.png" width="600" />
  </div>
</div>


---

# Essential libraries - 'The Scientific Stack'
You will almost always use these !

These are the most foundational and 'robust' libraries. There is overlap in some functions but generally:
<v-click>

1. **Essential Data Structures** 
	- [**numpy**](https://numpy.org/doc/stable/user/): replaces Python list, math library for vectorized N-dimensional maths
</v-click>

<v-click>

2. **Essential Scientific Computing Routines + Common Statistics**  
	- [**scipy**](https://docs.scipy.org/doc/scipy/tutorial/index.html): hypothesis tests, curve-fitting
</v-click>

<v-click>

3. **Data Visualization**  
	- [**matplotlib**](https://matplotlib.org/stable/gallery/index.html): can do nearly everything 2D+3D plotting but needs tweaking
	- [**seaborn**](https://seaborn.pydata.org/examples/index.html): wraps matplotlib to provide streamlined plotting of common scientific plots and colormaps
</v-click>

<v-click>

4. **Fundamental Image Processing**
	- [**pandas**](https://pandas.pydata.org/docs/reference/index.html): Streamlines reading and writing of table formats e.g. .csv, .txt, .tsv, .xlsx. Brings R and SQL table manipulation routines like merge to python. 
</v-click>

---

# Essential libraries - 'The Scientific Stack'
You will almost always use these !

<v-click>

5. **Fundamental Tabular Processing**
	- [**scikit-image**](https://scikit-image.org/docs/stable/auto_examples/): Modern image file reading/writing, no support for video like .avi, .mp4, common image processing routines with example code. Includes biological examples.
	- [**opencv**](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html): Industry-performant image processing developed in C, now made available for Python. Offers typically faster algorithms but documentation is difficult to understand, few examples, primarily for 2D and general computer vision.
</v-click>

<v-click>

6. **Fundamental Machine Learning**
	- [**scikit-learn**](https://scikit-learn.org/1.5/auto_examples/index.html): Classical machine learning algorithms with a single function API i.e. algorithm.fit() and algorithm.predict(). Extremely easy to use with many examples.	
		- Clustering, 
		- Data preprocessing e.g. Standard Scale, one-hot encoding 
		- Linear regression models e.g. OLS, Ridge, LASSO, ElasticNet; Random Forests; SVMs 
</v-click>


---

# Essential libraries - 'The Scientific Stack'
You will almost always use these !

These are the most foundational and 'robust' libraries. There is overlap in some functions but generally:  
- [**numpy**](https://numpy.org/doc/stable/user/):
	+ multidimensional arrays that perform fast, vectorized operations (replaces lists)
	+ masked arrays, array creation, linear algebra, summary statistics

<v-click>

```python
import numpy as np 
# python lists do not support vectorized processing
x = ['cats', 'dogs', 'humans', 'aliens']
y = [0,1,2,3,4,5]
# convert to numpy array
x_np = np.array(x); print(x_np.dtype) # this converts to a '<U6', 6-character string
y_np = np.array(y); print(y_np.dtype) # this converts to a np.int32, 32-bit integer
# we can do Boolean operations on all elements simultaneously
x_np_equal_1 = x_np == 'dogs' # return array same size as x_np
# we can compute mean and standard deviation with one function call, and handle NaN's
y_mean = np.nanmean(y_np) 
y_std = np.nanstd(y_np) 
```
</v-click>
---

# Essential libraries - 'The Scientific Stack'
You will almost always use these !

- [**scipy**](https://docs.scipy.org/doc/scipy/tutorial/index.html)
	+ Basic N-dimensional image processing e.g. correlation, Fourier Transforms, Distance Transforms
	+ Linear and non-linear curve-fitting methods
	+ Sparse linear algebra 
	+ Hypothesis Tests

<v-click>

```python
import numpy as np 
import scipy.stats as spstats

# generate two arrays of random numbers
x = np.random.normal(0,1,100)
y = np.random.uniform(0,1,100)

# Pearson's R
statistic, p_val, conf_internval = spstats.pearsonr(x,y)

# paired t-test
t_test_val, p_val, df, conf_interval = spstats.ttest(x,y)

```
</v-click>
---

# Essential libraries - 'The Scientific Stack'
You will almost always use these !
- [**matplotlib**](https://matplotlib.org/stable/gallery/index.html)
	+ <ins>THE</ins> plotting library (comparable to ggplot in R) 
	+ fully customizable plotting (require tweaking to get nice figures)

<v-click>

```python
import numpy as np 
import pylab as plt # pylab is a shortcut for matplotlib.pylab 

# generate two arrays of random numbers
x = np.random.normal(0,1,100)
y = np.random.uniform(0,1,100)
# create a figure and plot x vs y, layering different commands
plt.figure(figsize=(8,8))
plt.title('x vs y', fontname='Arial', fontsize=24)
plt.plot(x,y, 'o', ms=5, mec='k', mfc='w') # plot as white circles, black borders, markersize=5pt 
# further customizing plot appearance
plt.xlabel('x-axis', fontsize=18, fontname='Arial')
plt.ylabel('y-axis', fontsize=18, fontname='Arial')
plt.tick_params(right=True, length=10)
plt.savefig('myfigure.svg', dpi=300, bbox_inches='tight') # save as vector format, 300 dpi
plt.show() # tells python to display the figure

```
</v-click>

---

# Essential libraries - 'The Scientific Stack'
You will almost always use these !
- [**pandas**](https://pandas.pydata.org/docs/reference/index.html)
	+ <ins>THE</ins> data table library (comparable to R and SQL) 
	+ execute manipulations specifically geared for tables 
	+ interoperates with `numpy`, some libraries e.g. `seaborn` works best on tables.

<v-click>

```python
import numpy as np 
import pandas as pd

# generate two arrays of random numbers
x = np.random.normal(0,1,100)
y = np.random.uniform(0,1,100)
# stack them into a matrix that is table-like
tab_xy = np.vstack([x,y]).T 

# convert to pandas table
pandas_tab_xy = pd.DataFrame(tab_xy, columns=['x','y'], index=None) 
# save pandas table
pandas_tab_xy.to_csv('my_table.csv', index=None) # index=None means we don't get shifted table, unlike R
# convert to numpy 
numpy_tab_xy = pandas_tab_xy.values; columns_tab_xy = pandas_tab_xy.columns

```
</v-click>

---

# Essential libraries - 'The Scientific Stack'
You will almost always use these !
- [**scikit-image**](https://scikit-image.org/docs/stable/auto_examples/)
	+ Main easy-to-use image processing library with examples (common operations). 
	+ Slow, particularly 3D. Limited 3D image analysis.
	+ Definitely not complete! Start here, Google/chatGPT for more specific, advanced uses
	+ for Bioimaging Analysis, checkout this [book](https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/intro.html)


---

# Essential libraries - 'The Scientific Stack'
You will almost always use these !
- [**scikit-learn**](https://scikit-learn.org/1.5/auto_examples/index.html)


---

# Installing libraries


---

# How to get started with a new library?
Browse example use cases, read documentation


---

# Locating a function's API documentation
Locate example code snippets, then read function API documentation


---

# Beware: Libraries can have conflicting dependencies
One conda environment for each new project


---

# NumPy: Efficient data structures and processing for data analysis



