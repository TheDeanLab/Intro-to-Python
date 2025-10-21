
---
layout: center
---

# Introduction to Machine Learning

---

# Introduction to Machine Learning


Field of AI where computers learn patterns from data to make predictions or decisions.

<br>
<v-click>

- **Types**:
   - **Supervised Learning**: Uses labeled data to predict outcomes (e.g., Linear Regression, Decision Trees).
   - **Unsupervised Learning**: Finds patterns in unlabeled data (e.g., K-Means Clustering, PCA).

</v-click>
<br>
<v-click>

- **Classification vs. Regression**:
   - **Classification**: Predicts categories (e.g., spam vs. not spam).
   - **Regression**: Predicts continuous values (e.g., house prices).

</v-click>

---

# Introduction to scikit-learn

A powerful Python library for machine learning providing data preprocessing, modeling, and evaluation.

<v-click>

- **Core Features**:
   - Algorithms for classification, regression, clustering, and dimensionality reduction.
   - Tools for data splitting, cross-validation, and performance metrics.
   - Integrates easily with `NumPy` and `pandas`.

</v-click>
<v-click>

- **Strengths**:
   - Easy-to-use API with consistent syntax across models.
   - Extensive documentation and community support.
   - Great for rapid prototyping and experimentation.

</v-click>
<v-click>

- **Weaknesses**:
   - Not ideal for deep learning (use `TensorFlow` or `PyTorch` for that).
   - Limited support for handling very large datasets directly (consider `Dask` or `Spark`).

</v-click>

---

# Unsupervised Machine Learning

Unsupervised learning is used to find patterns or groupings in data without labeled outcomes.

<v-click>

### Clustering 
Group data into clusters based on similarity.
   - **K-Means**: Partitions data into a set number of clusters.
   - **Hierarchical Clustering**: Builds a tree of clusters for hierarchical structure.
   - **DBSCAN**: Identifies clusters based on density, useful for finding irregular shapes.

</v-click>
<br>
<v-click>

### Dimensionality Reduction 
Reduce data complexity, retain important structure.
   - **Principal Component Analysis (PCA)**: Reduces data to components that maximize variance.
   - **t-SNE**: Preserves local relationships in data, ideal for visualizing clusters.
   - **UMAP**: Maintains global and local structure, suitable for large datasets.

</v-click>

---

# K-Means Clustering

[sklearn.cluster.KMeans](https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html)

<div style="display: flex; justify-content: space-between;">
    <div>
        - Partitions data into K clusters by minimizing the distance from each point to its assigned cluster’s centroid.<br>
        - Iteratively adjusts cluster centers and assignments until convergence.
    </div>
    <div>
        <img src="https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif" width="200">
    </div>
</div>

<v-click>

# DBSCAN

[sklearn.cluster.DBSCAN](https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.DBSCAN.html)

<div style="display: flex; justify-content: space-between;">
    <div>
        - Groups dense areas of data points while marking outliers as noise.<br>
        - Effective for arbitrary-shaped clusters, resistant to noise in data.
    </div>
    <div>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/DBSCAN-Illustration.svg/300px-DBSCAN-Illustration.svg.png" width="200">
    </div>
</div>

</v-click>

---

# Principal Component Analysis (PCA)

[sklearn.decomposition.PCA](https://scikit-learn.org/1.5/modules/generated/sklearn.decomposition.PCA.html)

- Reduces dimensions by projecting data onto new axes that maximize variance.<br>
- Captures the most important features while minimizing data loss.

<br>
<v-click>

# t-SNE (t-Distributed Stochastic Neighbor Embedding)

[sklearn.manifold.TSNE](https://scikit-learn.org/1.5/modules/generated/sklearn.manifold.TSNE.html)

- Maps high-dimensional data into a 2D or 3D space, preserving local similarity.<br>
- Excellent for visualizing clusters, especially in high-dimensional data.

</v-click>
<v-click>

<center><img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/0*vz4K-fqiy19u_W-9" width="600"></center>

</v-click>

---

# Supervised Learning

Supervised learning uses labeled data to train models to make predictions or classifications based on input features.

<v-click>

### **Classification** 
Assigns labels to input data based on trained categories.
- **Common Techniques**: Logistic Regression, Support Vector Machines (SVM), Decision Trees.

</v-click>
<br>
<v-click>

### **Regression**
Predicts continuous outcomes based on input data.
- **Common Techniques**: Linear Regression, Ridge Regression, Random Forest Regression.

</v-click>

---

# Logistic Regression

[sklearn.linear_model.LogisticRegression](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html)

<div style="display: flex; justify-content: space-between;">
    <div>
        - Models the probability of a binary outcome using the logistic function.<br>
        - Provides interpretable coefficients, making it useful for binary classification problems.
    </div>
    <div>
        <img src="https://d1.awsstatic.com/S-curve.36de3c694cafe97ef4e391ed26a5cb0b357f6316.png" width="300">
    </div>
</div>

<v-click>

# Support Vector Machines (SVM)

[sklearn.svm.SVC](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVC.html)

<div style="display: flex; justify-content: space-between;">
    <div>
        - Finds the optimal hyperplane that maximizes margin between classes.<br>
        - Effective in high-dimensional spaces and robust to outliers.<br>
        - Linear or nonlinear based on kernel choice.
    </div>
    <div>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Kernel_Machine.svg/1920px-Kernel_Machine.svg.png" width="300">
    </div>
</div>

</v-click>

---

# Linear Regression

[sklearn.linear_model.LinearRegression](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html)

<div style="display: flex; justify-content: space-between;">
    <div>
        - Models the relationship between a dependent variable and one or more independent variables by fitting a linear equation.<br>
        - Simple yet powerful for predicting continuous outcomes.
    </div>
    <div>
        <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*LEmBCYAttxS6uI6rEyPLMQ.png" width="300">
    </div>
</div>

<v-click>

# Ridge Regression

[sklearn.linear_model.Ridge](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Ridge.html)

<div style="display: flex; justify-content: space-between;">
    <div>
        - A type of linear regression with L2 regularization to prevent overfitting.<br>
        - Shrinks the coefficients, improving model generalization.<br>
        - <b>Lasso</b> is similar with L1 regularization.
    </div>
    <div>
        <img src="https://i0.wp.com/thaddeus-segura.com/wp-content/uploads/2020/09/reg-intro.png?resize=768%2C421&ssl=1" width="300">
    </div>
</div>

</v-click>

--- 

# Example: Linear Regression with `scikit-learn`

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Generate some example data
X = np.linspace(0, 10, 101)      # Features (independent variable)
y = X + 2*np.sin(X) + np.random.randn(len(X))  # Target (dependent variable)
X = X[:,None]  # sklearn models expect N x M features
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

<v-click>

<div style="display: flex; justify-content: space-between;">
<div class="special-class">
```python
model = LinearRegression()       # Load the model
model.fit(X_train, y_train)      # Train the model

y_pred = model.predict(X_test)   # Make predictions
r2 = r2_score(y_test, y_pred)    # Evaluate the model

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='r')
plt.title(f"R-squared: {r2:.4f}")
plt.xlabel('X_test')
plt.legend(('y_test', 'y_pred'))
```
</div>
<div>
<img src="/images/linear_reg.png" width=300>
</div>
</div>

</v-click>

--- 

# Data Preparation for Machine Learning

- **Preparation for Classification vs. Regression**:
   - **Regression**: Target values are continuous; consider scaling features (e.g., standardization).
   - **Classification**: Ensure target labels are categorical; features may need encoding (e.g., <u>one-hot</u>).

<v-click>

<table><tbody><tr>
<td>

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Example: one-hot encoding class labels for a classification task
labels = np.array(['cat', 'dog', 'rabbit', 'dog', 'cat'])
encoder = OneHotEncoder(sparse=False)
encoded_labels = encoder.fit_transform(labels.reshape(-1, 1))

# Print the result
print("Original labels:", labels)
print("One-hot encoded labels:\n", encoded_labels)
```

</td>
<td>

```console
Original labels: ['cat' 'dog' 'rabbit' 'dog' 'cat']
One-hot encoded labels:
 [[1. 0. 0.]
  [0. 1. 0.]
  [0. 0. 1.]
  [0. 1. 0.]
  [1. 0. 0.]]
```

</td>
</tr></tbody></table>

</v-click>
<v-click>

- <b>Dimensionality reduction</b> from the previous slide could be considered data preparation for ML!

<center><b><u>
Visualize your data!
</u></b></center>

</v-click>

---

# Data Preparation for Machine Learning

<br>
<v-click>

- **Features vs. Targets**:
   - **Features**: Input variables (e.g., age, income) that the model uses for learning.
   - **Targets**: Output variable(s) the model predicts (e.g., class label for classification, a numeric value for regression).

</v-click>
<br>
<v-click>

- **Train/Test/Validation Split**:
   - **Train Set**: Used for training the model.
   - **Validation Set**: Used for tuning model hyperparameters.
   - **Test Set**: Used to evaluate model performance on unseen data.

</v-click>

---

# Train/Test/Validation Split with `scikit-learn`

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Example data: 100 samples with 5 features each
X = np.random.rand(100, 5)  # Features
y = np.random.randint(0, 2, 100)  # Binary target (0 or 1)

# Step 1: Split the data into 80% training and 20% temporary data (for validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Split the temporary data into 50% validation and 50% test (10% of original each)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print the sizes of each set
print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)
print("Test set size:", X_test.shape)
```

```console
Training set size: (80, 5)
Validation set size: (10, 5)
Test set size: (10, 5)
```

---
layout: center
---

# Activity
.\activities\activity_python-scientific-computing.ipynb

We will use the tools learned in this section to do:
- Image Processing
- Statistical Analysis
- Machine Learning