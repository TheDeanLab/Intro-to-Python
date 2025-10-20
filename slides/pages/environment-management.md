---
layout: center
---

# Dependency Management and Environment Requirements
## Clean, Reproducible Code 


---
layout: image-right
image: /images/environments.png
backgroundSize: contain
---

# Environment Management with Anaconda
Creating Reproducible Development Environments

** **

- An environment is like a tissue culture flask. 
- It guarantees that the software in one project doesn't interfere with another. 
- This enables a stable and reproducible space for your code.

---

# Environment Management with Anaconda
Why Does it Matter?

** **

<v-click>

**Version Inconsistencies.** 
  - Python libraries and tools are constantly evolving. 
  - Different projects might require different versions of the same library, leading to conflicts and unexpected behavior.
</v-click>

<v-click>

**Reproducibility.**
  - For scientific computing and data analysis tasks, it's crucial to reproduce results. 
  - This is challenging without a consistent environment, especially when sharing work with peers or publishing results.
</v-click>





---

# Environment Management with Anaconda
Why Does it Matter?

** **
<v-click>

**Ease of Sharing.**
  - With a well-managed environment, developers can easily share their projects, ensuring that others can run their code without stumbling upon missing dependencies or version issues.
</v-click>

<v-click>

**Isolation.**
  - Keeping project environments separate ensures that specific dependencies or version requirements of one project don't interfere with another, leading to cleaner and more stable software.
</v-click>


---

# Environment Management with Anaconda
Miniconda vs Anaconda
** **
<v-click>

**Size and Content.**
  - Anaconda is a large distribution that comes pre-loaded with over 1500 packages tailored for scientific computing, data science, and machine learning. 
  - Miniconda, on the other hand, is a minimalistic distribution, containing only the package manager (conda) and a minimal set of dependencies. 
  - Due to its bundled packages, Anaconda requires more disk space upon installation compared to Miniconda.
</v-click>

<v-click>

**Flexibility vs. Convenience.**
  - While Anaconda provides an out-of-the-box solution with a wide array of pre-installed packages, Miniconda offers flexibility by allowing users to install only the packages they need, helping to keep the environment lightweight.
</v-click>

---

# Environment Management with Anaconda
What is a Package?
** **

<v-click>

**Software Collection.**
  - A package is a bundled collection of software tools, libraries, and dependencies that function together to achieve a specific task or set of tasks.

</v-click>

<v-click>

**Version Management.**
  - Each package has specific versioning, allowing users to install, update, or rollback to particular versions as needed, ensuring compatibility and stability in projects.

</v-click>

<v-click>

**Dependency Handling.**
  - When a package is installed in Anaconda, the system automatically manages and installs any required dependencies, ensuring seamless functionality and reducing manual setup efforts.

</v-click>


---

# Environment Management with Anaconda
Creating a New Environment

** **

<v-click>

**Create a new environment.**

```bash
conda create --name EnvironmentName
```

</v-click>

<v-click>

**Create a new environment with specific Python version**

```bash
conda create --name EnvironmentName python=3.10
```

</v-click>

<v-click>

**Create a new environment from a YAML or text file.**

```bash
conda create --name EnvironmentName file=package_contents.yml
conda create --name EnvironmentName file=package_contents.txt
```

</v-click>

---
layout: image-right
image: /images/environments2.png
backgroundSize: contain
---

# Environment Management with Anaconda
Creating New Environments from Text Files

** **

Each package, and all of its dependencies, explicitly imported from a package manager (pip, conda, etc.)

**Provides:**
- Version Control (e.g., pyserial==3.5)
- Platform Control (e.g., sys_platform == “darwin”)


---
layout: image-right
image: /images/conda_list.png
backgroundSize: contain
---

# Environment Management with Anaconda
Working with Environments
** **

**List all environments.**

```
conda env list
```

**Activate an environment**

```
conda activate EnvironmentName
```

**List Environment Packages.**

```
conda list
```



---

# Adding your environment to a JupyterLab Notebook

1. **Activate Your Environment:**
    - Open your terminal or command prompt and activate the environment you want to add to Jupyter:
      ```bash
      conda activate environment_name
      ```

2. **Install the `ipykernel` Package (if not already installed):**
    - Ensure that `ipykernel` is installed in your environment so it can be used as a Jupyter kernel:
      ```bash
      pip install ipykernel
      ```

---

# Adding your environment to a JupyterLab Notebook

1. **Add the Environment as a JupyterLab Kernel:**
    - Use the following command to add your environment as a kernel in Jupyter:
      ```bash
      python -m ipykernel install --user --name environment_name --display-name "My Environment"
      ```
    - Replace `"environment_name"` with your environment’s name, and you can customize the display name shown in Jupyter (e.g., "My Environment").

2. **Verify in JupyterLab:**
    - Open Jupyter Notebook or JupyterLab:
      ```bash
      jupyter notebook
      ```
    - In a new notebook, go to **Kernel > Change Kernel**, and you should see your environment.


---

# Environment Management with Anaconda
Working with Environments
** **

**Important:**
- Do not mix package managers (e.g., conda and pip).
- Be judicious and explicit with your dependencies.

---
layout: center
---

# Activity
Create a Python Environment and Install Packages

---

# Activity: Create a Python Environment and Install Packages
Create a Python Environment and Install Packages

### Objective:
- Create a new environment with a specific Python version.
- Activate the environment.
- Install dependencies from a text file.

---

# Activity: Create a Python Environment and Install Packages
Create a Python Environment and Install Packages

### Steps:

1. **Create a New Environment with Python 3.9:**
    - Open your terminal or command prompt.
    - Run the following command to create a new environment with Python 3.9:
      ```bash
      conda create --name myenv python=3.9
      ```

2. **Activate the New Environment:**
    - Once the environment is created, activate it using the following command:
      ```bash
      conda activate myenv
      ```

---

# Activity: Create a Python Environment and Install Packages
Create a Python Environment and Install Packages

3. **Create a `requirements.txt` File:**
    - In your working directory, create a new text file called `requirements.txt`.
    - Add a few packages to the file, for example:
      ```
      numpy
      pandas
      matplotlib
      ```

4. **Install Packages from the `requirements.txt` File:**
    - Use the following command to install the packages listed in the `requirements.txt` file:
      ```bash
      pip install -r requirements.txt
      ```

---

# Activity: Create a Python Environment and Install Packages
Create a Python Environment and Install Packages

5. **Verify the Installation:**
    - After installation, verify that the packages were installed correctly by running:
      ```bash
      conda list
      ```

6. **Import Dependencies and Execute Logic:**
    - Create a Python script called `plot_image.py`.
    - Import numpy and matplotlib.
    - Create a random 2D matrix that is 512 x 512 pixels.
    - Plot the original image.
    - Manipulate the image (e.g., taket he inverse).
    - Plot the manipulated image.
