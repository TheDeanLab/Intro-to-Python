---
layout: center
---

# Data Visualization
## Making publication-ready figures

---

# Matplotlib 
(the primary image plotting library)

- Matplotlib is a **beast** of a library and it is impossible to cover all functionalities
- Fortunately there is an extensive [examples gallery](https://matplotlib.org/stable/gallery/index), and [cheatsheets](https://matplotlib.org/cheatsheets/) 

<v-click>

**Basic Usage**
- Figure with single plot
```python
plt.figure() # create a figure canvas
plt.title() # give the figure a name
plt.imshow() # display a figure
plt.scatter() # plot some scattered points
plt.savefig() # save the figure (the extension specifies the image format)
plt.show() # display the figure
```
</v-click>

<v-click>

- Figure with multiple subplots
```python
fig, ax = plt.subplots(nrows=2, ncols=3) # create a figure canvas, equally split 2 rows, 3 columns
ax[0,0].imshow() # display image in 1st row, 1st col
ax[0,1].plot() # plot line in 1st row, 2nd col
ax[1,1].bar() # plot bar graph in 2nd row, 1st col
```
</v-click>

---

# Matplotlib 
(the primary image plotting library)

**Common plotting commands for science**:

<v-click>

- line plot: `plt.plot`, (also used instead of scatter if marker is set) 
```python
plt.plot(x,y,'g.-',lw=3) # plot x,y as a line
```
</v-click>

<v-click>

- bar plot: `plt.bar` (plot vertical bars), `plt.hbar` (plot horizontal bars)
```python
plt.bar(x,y,width=1) # plot vertical bar at x, width of 1
```
</v-click>

<v-click>

- error bar plot: `plt.errorbar`
```python
plt.bar(x,y,xerr=x_errors, yerr=y_errors) # plot x,y as a line with error bars for each
```
</v-click>


<v-click>

- box plot: `plt.boxplot` and violin plot: `plt.violinplot`
```python
plt.boxplot([[data_group_1],[data_group_2],[data_group_3]]) # boxplot of values from 3 groups
plt.violinplot([[data_group_1],[data_group_2],[data_group_3]]) # boxplot of values from 3 groups
```
</v-click>

<v-click>

- display 2D image: `plt.imshow` or `plt.matshow`
```python
plt.imshow(img, cmap='Greys') # display image and use gray colormap
```
</v-click>

--- 
layout: image-right
image: /images/matplotlib_anatomy.png
backgroundSize: contain
---

# Matplotlib
(Anatomy of a matplotlib figure)

- `fig = plt.figure()`: global plot canvas
- `ax = fig.add_axes([0.2, 0.17, 0.68, 0.7], aspect=1)`: plotting grid 

Basic concepts:

- title: title
- spine: outer border
- grid: gridlines
- ticks: major/minor annotation of x-, y-axis
- label: name of x-, y- axis
- legend: plot legend

See: [matplotlib figure anatomy](https://matplotlib.org/stable/gallery/showcase/anatomy.html)


---

# Matplotlib
Pros and Cons

<v-click>

- **Pros**:
	- Customizable - users have absolute control over every element
	- easy-to-install (i.e. always works, no complex dependencies)
	- many plot types with a function, working directly on numpy arrays
	- extensive developer support (longest developed, and actively developing)
	- extensive documentation and examples
	- export in many formats including vector graphics (.svg, .pdf)
</v-click>

<v-click>

- **Cons**:
	- needs tweaking of figure settings to look good
	- not all scientific plots supported or need lots of customizing (or do it in illustrator)
	- limited support for interactive and animated plots
</v-click>

--- 
layout: image-right
image: /images/seaborn.png
backgroundSize: contain
---

# Matplotlib Alternative: seaborn
Rapid prototyping of general scientific figures 

<v-click>

- Install
```python
pip install seaborn
conda install seaborn
```
</v-click>

<v-click>

- Provide data as `pandas.DataFrame`
</v-click>

<v-click>

- Choose plot from gallery and reuse code
```python
import seaborn as sns
violin = sns.violinplot(data=my_table, x='x_var_name', y='y_var_name')
```
</v-click>

<v-click>

- Generate and use [color palettes](https://seaborn.pydata.org/tutorial/color_palettes.html)
```python
color = sns.color_palette("husl", 8) # give me 8 colors from "husl" colormap
sns.palplot(color) # visualize the palette
```
</v-click>

--- 

# Matplotlib Alternative: anndata, scanpy and scverse
Scientific data structures and publication-ready scientific figures  

- **[scverse](https://scverse.org/packages/#core-packages)**: 

Python ecosystem for single-cell transcriptomics and spatial transcriptomics analysis

<div class="flex flex-col items-center">
  <div>
    <img src="/images/scverse.png" width="700" />
  </div>
</div>

--- 
layout: image-right
image: /images/anndata_schema.png
backgroundSize: contain
---

# Matplotlib Alternative: anndata, scanpy and scverse
Scientific data structures and publication-ready scientific figures  

- **[anndata](https://anndata.readthedocs.io/en/latest/)**: a data structure for science 
```python
pip install anndata
```

- an `anndata` object is a giant dictionary. This means you use 1 variable name for all your data, metadata and results (which are all dictionary items). This is all saved into a single `h5ad` file


--- 

# Matplotlib Alternative: anndata, scanpy and scverse
Scientific data structures and publication-ready scientific figures  

<v-click>

- **[scanpy](https://scanpy.readthedocs.io/en/stable/)**: Python library for single cell transcriptomics analysis (all standard analyses)
```python
pip install scanpy
```
</v-click>

<v-click>

- Publication-ready [one line plotting](https://scanpy.readthedocs.io/en/latest/tutorials/plotting/core.html) of `anndata` object e.g.

<div class="flex flex-col items-center">
  <div>
    <img src="/images/scanpy_plotting_example.png" width="700" />
  </div>
</div>
</v-click>

<v-click>

- [Other plots](https://scanpy.readthedocs.io/en/latest/tutorials/basics/clustering.html) tied to computational step e.g. UMAP, clustering, graphs
</v-click>


--- 

# Interactive plotting in Python: `Bokeh`
Interactive plots in the browser that respond to user input

- [`Bokeh`](https://docs.bokeh.org/en/latest/docs/gallery.html) generates interactive plots in the web browser
```python
pip install bokeh
```

<div class="flex flex-col items-center">
  <div>
    <img src="/images/bokeh.png" width="600" />
  </div>
</div>


---

# Tip of the iceberg: countless possibilities
Many choices, once you have the data

<div class="flex flex-col items-center">
  <div>
    <img src="/images/viz_landscape.png" width="500" />
    	<v-click>
		Save your result, and you can use any you like!
		</v-click>
  </div>
</div>


---

# How to save my data and results?
General saving options

- arbitrary python structures: `pickle`
```python
import pickle

savedict={'save_1': my_dict, 'save_2':[list], 'save_3': numpy_array, 'save_4':'string'}

with open(savepicklefile, 'wb') as f:
    pickle.dump(savedict, f)
with open(savepicklefile, 'rb') as f:
	data = pickle.load(f)
```

- numpy arrays: `numpy.save` and `numpy.load`
```python
with open('test.npy', 'wb') as f:
	np.save(f, np.array([1, 2]))
with open('test.npy', 'rb') as f:
	data = np.load(f)
```

---
layout : center
---

# Activity
Visualize properties of patient dataset

(go to Jupyter notebook in the GitHub : activity_python-data_visualization.ipynb)
