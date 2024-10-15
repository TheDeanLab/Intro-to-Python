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

# stack the two random number arrays into a matrix
tab_xy = np.vstack([x,y]).T 

# convert to pandas table
pandas_tab_xy = pd.DataFrame(tab_xy, columns=['x','y'], index=None) 
# save pandas table
pandas_tab_xy.to_csv('my_table.csv', index=None) # index=None means we don't get shifted table, unlike R
# convert to numpy 
numpy_tab_xy = pandas_tab_xy.values; columns_tab_xy = pandas_tab_xy.columns

# read in a .csv as a pandas DataFrame if you have one
table = pd.read_csv('my_table.csv')
```
</v-click>

---

# Essential libraries - 'The Scientific Stack'
You will almost always use these !
- [**scikit-image**](https://scikit-image.org/docs/stable/auto_examples/)
	+ Main easy-to-use image processing library with examples (common operations). 
	+ Slow, particularly 3D. Limited 3D image analysis.
	+ Definitely not complete! Start here, Google/chatGPT for more specific, advanced uses
	+ for Bioimaging Analysis, checkout this [book](https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/intro.html). See [bio-formats](https://docs.openmicroscopy.org/bio-formats/6.8.0/developers/python-dev.html) for more general handling of microscopy image formats

<v-click>

```python
import skimage.io as skio
import skimage.transform as sktform
import pylab as plt 
# read common image formats e.g. .png, .tif, .jpg. For .nd2 (install nd2), .czi (install czifile)
img = skio.imread('my_image.tif')
# save image
skio.imsave('my_saved_image.tif', img)
# resize image by linear interpolation
img_resize = sktform.resize(img, output_shape=(256,256), order=1, preserve_range=True) 
# view image, figsize controls resoluton on-screen
plt.figure(figsize=(10,10)); plt.imshow(img_resize, cmap='gray'); plt.show()
```
</v-click>


---

# Essential libraries - 'The Scientific Stack'
You will almost always use these !
- [**scikit-learn**](https://scikit-learn.org/1.5/auto_examples/index.html)
	- Simplest library for machine learning. Every function has the same way of using. 
	- Contains useful data preprocessing routines, and toy datasets
	- Primarily, a tour of classical machine learning techniques, which deliver very good performance and further improved with parameter tuning using `auto-sklearn` [library](https://automl.github.io/auto-sklearn/master/)

<v-click>

```python
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pylab as plt 

iris = datasets.load_iris() # load iris dataset, this has 3 features/dimensions
iris_sdscale = StandardScaler().fit_transform(iris.data) # standard scale each feature to be mean=0, std=1
pca_iris_sdscale = PCA(n_components=2).fit_transform(iris_sdscale) # use PCA to map to 2 dimensions to plot
# plot, and color points by the 3 iris types, and choose a colormap
plt.figure(figsize=(8,8)) 
plt.scatter(pca_iris_sdscale[:,0], pca_iris_sdscale[:,1], c=iris.target, colormap='Spectral')
plt.show()
```
</v-click>

---

# Installing libraries

- All these libraries can be installed together using `conda` or `pip`. 
	- `conda`:
		```python
		conda install numpy scipy matplotlib pandas scikit-image scikit-learn 
		```
	- `pip`:
		```python
		pip install numpy scipy matplotlib pandas scikit-image scikit-learn 
		```

<v-click>

- Beware:
	- Python packages may have complex and conflicting dependencies 
		- e.g. PyQT (required by many GUIs)
	- Python packages may require compilation if not pre-compiled for OS 
	- Best practice:
		- work in mainstream, stable Linux OS e.g. Ubuntu
		- create new conda environment for each project
		- install minimal set of libraries jointly upfront using e.g. `requirements.txt`
</v-click>

---

# 50% of time: File and folder manipulation

<v-click>

- **Real data is heterogeneous:** collected in different ways, different data formats
</v-click>

<v-click>

- **Every user is different:** different file naming, different folder stuctures (and different between experiments)
</v-click>

<v-click>

- **Operating systems are different:** different filepath convention
	- Windows: r"C:\Work\Projects\Intro-to-Python-2024\slides"
	- Windows: "C:\\\\Work\\\\Projects\\\\Intro-to-Python-2024\\\\slides"
	- UNIX: "/home/Work/Projects/Intro-to-Python-2024/slides"
</v-click>

<v-click>

- <ins>Most of the time</ins> is spent on locating the necessary files, reading and parsing them for the desired data in order to perform analysis:
	- **First 50%** is finding and reading the right files in the right folders into data structures (Built-in Python libraries)
	- **Second 50%** is writing code to extract the data in the right format to use for analysis (Day 1 + libraries)
</v-click>


---
layout: image-right
image: /images/glob_example.png
backgroundSize: 60%
---

# 50% of time: File and folder manipulation
`glob` library for file & folder search

- [`glob.glob`](https://docs.python.org/3/library/glob.html) enables finding files and folders matching a pattern. 
- pattern is given by [regular expression](https://www.geeksforgeeks.org/write-regular-expressions/#) (regex) e.g. `*` denotes wildcard  
<div class="flex flex-col items-center">
  <div>
    <img src="/images/glob_example_script.png" width="700" />
  </div>
</div>

- `recursive=True` searches also subfolders
- returns list of unordered file or folder paths 
- use `sorted()` or `numpy.sort()` to sort

---

# 50% of time: File and folder manipulation
`os` library for operating system (OS) operations

<v-click>

- Get current working directory (`os.getcwd`) and change to new working directory (`os.chdir`)
```python
import os
cwd = os.getcwd()
os.chdir('/path/to/new/directory')
```
</v-click>

<v-click>

- Create a path from strings (`os.path.join`)
```python
os.path.join('prefix/path', 'middle/path', 'suffix/path')
```
</v-click>

<v-click>

- Creating a directory (`os.mkdir`) and all intermediate directories (`os.mkdirs`)
```python
os.mkdir('/path/to/new/directory')
os.makedirs('/path/to/new/directory/with/intermediate/directories')
```
</v-click>


<v-click>

- Checking if path is a folder (`os.path.isdir`) or file (`os.path.isfile`)
```python
os.path.isdir('/path/to/directory')
os.path.isfile('/path/to/file')
```
</v-click>


---

# How to get started with a (new) library?
Look for examples and demos, follow-up with function documentation

<v-click>

1. Libraries typically all come with example scripts. Make sure to run to:
	- check library is correctly installed
	- check function usage
</v-click>

<v-click>

2. Some libraries have user guides and notebooks with detailed explanations, like a book
	- read them to understand topics - these are way more useful than a paper!
</v-click>

<v-click>

3. Check documentation of function for parameters, what format should the input be, what does it return
	- In `spyder`, using `ctrl-left click` automatically takes you to a function's source code
	- Access docstring with `help(function)` or online published function API references
</v-click>

<v-click>

4. Ask in relevant online general- and topic- specific forums:
	- [StackOverflow](https://stackoverflow.com/): general computing
	- [image.sc](https://forum.image.sc/): scientific image analysis (biological mainly)
	- ChatGPT / Gemini AI and Google search engines (search online)
</v-click>

<v-click>

5. (Most important) Practice, read lots of code and documentation
</v-click>


---

# Essential Skill: debugging code
There are only two types of error: 1) code syntax and 2) logic error

I am getting errors, it doesn't seem to work - how do we detect and fix? Here are some essential tools and tips.

<v-click>

-  `print()`: 
	- the simplest debugging tool, print your variables - are they what you expect?
</v-click>

<v-click>

-  `assert()`: 
	- do you have a variable that you know should be a particular value? Use `assert` which act like a brake, stopping the code when the condition is not met 
</v-click>

<v-click>

-  `help()`: 
	- everything in Python is an 'object'. you can use help() to anything to get more information, even the variables you create or assign! 
</v-click>

<v-click>

-  **Plan your logic first:** what are the steps ? breakdown further to that you know how to do : 
</v-click>

<v-click>

-  **Construct and solve a toy problem:** check usage of a function or method 
</v-click>
---
layout : center
---

# Activity
Use libraries to do exploratory analysis of patient dataset

(go to Jupyter notebook XXX)