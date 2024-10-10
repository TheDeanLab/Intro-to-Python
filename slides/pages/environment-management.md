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
**Creating Reproducible Development Environments**

** **

- An environment is like a tissue culture flask. 
- It insures that the software in one project doesn't interfere with another. 
- This enables a stable and reproducible space for your code.

---

# Environment Management with Anaconda
**Why Does it Matter?**

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
**Why Does it Matter?**

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
**Miniconda vs Anaconda**
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
**What is a Package?**
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
**Creating a New Environment**
** **

<v-click>

**Create a new environment.**

```
conda create --name EnvironmentName
```

</v-click>

<v-click>

**Create a new environment with specific Python version**

```
conda create --name EnvironmentName python=3.10
```

</v-click>

<v-click>

**Create a new environment from a YAML or text file.**

```
conda create --name EnvironmentName file=package_contents.yml
```

</v-click>



---
layout: image-right
image: /images/environments2.png
backgroundSize: contain
---

# Environment Management with Anaconda
**Working with Environments**
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
layout: image-right
# image: /images/environments2.png
backgroundSize: contain
---

# Environment Management with Anaconda
**Creating New Environments from Text Files**
** **

Each package, and all of its dependencies, explicitly imported from a package manager (pip, conda, etc.)*
Version Control 
(e.g., pyserial==3.5)
Platform Control
(e.g., sys_platform == “darwin”)


