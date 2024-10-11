---
layout: center
---

# Functions and Modules 

---

# Introduction to Functions in Python
What are Functions?

** **

- **What are Functions?**
  - A function is a block of organized, reusable code that performs a specific task.
  - Functions allow you to encapsulate code logic, making it easier to call the same code multiple times without rewriting it.
  - They are defined using the `def` keyword followed by the function name and parentheses `()`.

- **Example of a Simple Function:**
    ```python
    def greet(name):
        print(f"Hello, {name}!")
    
    greet("Alice")
    ```

---

# Introduction to Functions in Python
What are Functions?

** **
- **Benefits of Using Functions:**
  - **Code Reusability:**
    - Functions allow you to write code once and reuse it wherever needed, avoiding repetition.
  
  - **Modularity:**
    - By breaking down complex problems into smaller functions, you make your code more organized and easier to understand.

  - **Maintainability:**
    - Functions allow you to easily update or debug parts of your code without affecting the rest of the program.

---

# Introduction to Functions in Python
What are Functions?

** **

  - **Readability:**
    - Well-named functions make the code more readable by summarizing what specific sections of the code do.

  - **Avoiding Global Variables:**
    - Functions help avoid the misuse of global variables by encapsulating variables locally within the function.



---

# Introduction to Functions in Python
Functions with Return Values

** **

- **What are Functions?**
  - In addition to performing tasks, functions can return data back to the caller using the `return` statement.
  - This allows functions to output a result or value that can be used elsewhere in your code.
  - After a `return` statement, the function stops execution and passes the result back.

- **Example of a Function that Returns a Value:**
    ```python
    def add(a, b):
        return a + b
    
    result = add(3, 4)
    print(result)  # Output will be 7
    ```

---

# Introduction to Functions in Python
Functions with `*args` in Python

** **
- **What are `*args` in Functions?**
  - `*args` allows you to pass a variable number of positional arguments to a function.
  - The `*` operator collects all extra positional arguments passed to the function and bundles them into a tuple, making it easy to iterate over them or process multiple inputs.
  - This makes functions more flexible, allowing them to accept varying numbers of arguments without needing to be rewritten.

  ```python
  def add_all(*args):
      return sum(args)
  
  result = add_all(1, 2, 3, 4)
  print(result)  # Output will be 10

  result = add_all(1, 2)
  print(result)  # Output will be 3
  ```


---

# Introduction to Functions in Python
Functions with `**kwargs` in Python

** **

- **What are `**kwargs` in Functions?**
  - `**kwargs` allows you to pass a variable number of keyword arguments to a function.
  - The `**` operator collects these arguments into a dictionary.
  - Functions then accept a varying number of keyword arguments without needing predefined parameters.

  ```python
  def print_info(**kwargs):
      for key, value in kwargs.items():
          print(f"{key}: {value}")

  print_info(name="Alice", age=30)
  # Output:
  # name: Alice
  # age: 30
  ```

---
layout: center
---

# Documenting Your Functions

---

# Documenting Your Functions
Numpydoc

** **

- **What is Numpydoc?**
  - Numpydoc is a docstring format used for documenting Python code, particularly in scientific computing and data science libraries.
  - It’s widely used in projects like NumPy, SciPy, and other scientific Python packages.

- **Structure of a Numpydoc Docstring:**
  - Numpydoc follows a specific structure with defined sections to make documentation clear and consistent.
  - Common sections include:
    - **Parameters:** Describes the function's inputs.
    - **Returns:** Explains the output of the function.
    - **Raises:** Lists exceptions that the function can raise.
    - **Examples:** Provides usage examples to demonstrate how the function is used.

---

# Documenting Your Functions
Numpydoc

** **

- **Example of a Numpydoc Docstring:**
    ```python
    def add_numbers(a, b):
        """ Add two numbers.

        Parameters
        ----------
        a : int
            The first number.
        b : int
            The second number.

        Returns
        -------
        int
            The sum of the two numbers.

        Examples
        --------
        >>> add_numbers(3, 4)
        7
        """
        return a + b
    ```

--- 

# Documenting Your Functions
Numpydoc

** **

- **Why Use Numpydoc?**
  - Provides structured, readable documentation that is easy to follow.
  - Helps automate documentation generation for large projects.
  - Widely recognized in the scientific Python community, promoting consistency.


---

# Documenting Your Functions
Inline Type Hinting

** **

- **What is Inline Type Hinting?**
  - Inline type hinting allows you to specify the expected types of function arguments and return values directly in the function signature.
  - Introduced in Python 3.5, type hints improve code readability and help developers understand what types are expected.
  - Type hinting does not enforce type checking but provides a form of documentation for your code.

- **Example of a Function with Type Hinting:**
    ```python
    def add_numbers(a: int, b: int) -> int:
        return a + b
    ```

---

# Documenting Your Functions
Inline Type Hinting

** ** 

- **Benefits of Using Inline Type Hinting:**
  - **Improved Readability:**
    - Developers can quickly understand what types are expected for arguments and return values.
  
  - **Better Tooling:**
    - IDEs and code editors can offer better auto-completion, static analysis, and error detection based on type hints.
  
  - **Facilitates Collaboration:**
    - Team members working on the same codebase can understand function signatures at a glance, improving collaboration and reducing bugs.


---
layout: center
---

# Modules

---

# Modules
What are Modules?

** **

- **Definition:** 
  - A module is a Python file containing Python code (functions, variables, classes) that can be reused in other programs.
  
- **Purpose:**
  - Modules allow you to break your code into smaller, reusable, and manageable parts, promoting modular programming.


Modules help organize code and allow for code reuse across multiple programs.

---

# Modules
Built-In Modules

** **

- **The `import` Statement:**
    ```python
    import math
    print(math.sqrt(16))  # Outputs: 4.0
    ```

- **Selective Imports (`from module import`):**
    ```python
    from math import sqrt
    print(sqrt(16))  # Outputs: 4.0
    ```

- **Aliases (`import module as`):**
    ```python
    import numpy as np
    print(np.array([1, 2, 3]))
    ```


Python allows flexible importing: full modules, specific parts, or using aliases for convenience.

---

# Modules
Built-In Modules

** **

- Python comes with many useful built-in modules. Some examples include:

  - **math:** For mathematical operations.
  - **os:** For interacting with the operating system.
  - **random:** For generating random numbers.

- Example:
    ```python
    import random
    print(random.randint(1, 10))  # Outputs a random number between 1 and 10
    ```



---
layout: image-right
image: images/python_module_index.png
---

# Modules
Built-In Modules

** **

- Built-in modules provide essential functionality without requiring external installation. 
- A list of built-in modules available to you can be found at https://docs.python.org/3/py-modindex.html

<v-click>

<Arrow x1="420" y1="17" x2="480" y2="17" />
</v-click>



---

# Modules
Creating Custom Modules

** **

- **Creating a Module:**
    - You can create your own module by writing Python code in a `.py` file.

- **Example:**
    - Create a file `my_module.py`:
      ```python
      def greet(name):
          return f"Hello, {name}!"
      ```

    - Use the module in another script:
      ```python
      import my_module
      print(my_module.greet("Alice"))  # Outputs: Hello, Alice!
      ```


Custom modules allow you to structure your code and reuse it across different programs.

---

# Modules
The Importance of `__init__.py`

** **

- **What is `__init__.py`?**
  - `__init__.py` is a special Python file used to mark a directory as a Python package.
  - Without this file, Python will not treat the directory as a package, and the modules inside the directory cannot be imported.
  - When a directory contains an `__init__.py` file, you can import modules from the directory like a package.
  - The file can be empty or contain initialization code for the package.

---

# Modules
The Importance of `__init__.py`

** **

- **Basic Example of Package Structure:**
    ```
    my_package/
        __init__.py
        module1.py
        module2.py
    ```

    - You can import modules from the package:
      ```python
      from my_package import module1
      ```

`__init__.py` is be default empty, but can also be used to intialize package-level variables, import additional submodules, etc.

`__init__.py` is essential for structuring Python packages, enabling module imports, and controlling package initialization behavior.

---

# Modules
Exploring Module Search Path (`sys.path`)

** **


- **How Python Finds Modules:**
  - Python searches for modules using the paths listed in `sys.path`.

- **Checking `sys.path`:**
    ```python
    import sys
    print(sys.path)
    ```

- **Modifying `sys.path`:**
  - You can add directories to `sys.path` to include custom modules from other locations:
    ```python
    sys.path.append('/path/to/custom/modules')
    ```

Python's module search path (`sys.path`) determines where Python looks for modules and can be modified to include custom directories.



---

# Modules
Third-Party Modules and PyPI

** **

- **Installing Modules via `pip`:**
    - You can install external Python packages using `pip`:
      ```bash
      pip install requests
      ```

- **Popular Third-Party Libraries:**
    - `requests`: For making HTTP requests.
    - `numpy`: For numerical computing.
    - `pandas`: For data analysis.

- **PyPI (Python Package Index):**
    - PyPI is the repository for sharing and downloading Python packages.


Third-party modules from PyPI extend Python’s functionality, providing solutions for many domains.



---

# Modules
Understanding `if __name__ == "__main__"`

** **

- **What is `__name__`?**
  - In Python, `__name__` is a special built-in variable that represents the name of the current module.
  - When a Python file is run directly, `__name__` is set to `"__main__"`. However, when a file is imported as a module, `__name__` is set to the module's name instead.

- **Why use `if __name__ == "__main__":`?**
  - This statement ensures that a block of code is only executed when the script is run directly, not when it’s imported as a module in another script.
  - It is commonly used to encapsulate the "entry point" of a Python script.

---

# Modules
Understanding `if __name__ == "__main__"`

** **

- **Example:**
    ```python
    # myscript.py
    def main():
        print(5+5)
        
    if __name__ == "__main__":
        main()
    ```

- **Behavior:**
  - **When run directly:** The `main()` function will execute.
  - **When imported by another file:** The `main()` function will not automatically execute. 


It allows for reusable code by preventing specific parts of a script from running when the script is imported as a module elsewhere.


---

# Modules
Benefits of Using Modules

** **

- **Code Reusability:**
  - Reuse code across multiple programs without rewriting it.
  
- **Organized Codebase:**
  - Keep code clean and organized by dividing it into smaller modules.

- **Maintainability:**
  - Easier to maintain and update your codebase with modular components.


Modules improve code organization, reusability, and maintainability, making it easier to manage larger projects.

---
layout: center
---

# Activity
Create and Use Your Own Python Module


---

# Activity
Create and Use Your Own Python Module

** **

Goals
- Create a Python module with two simple functions.
- Import the module into another script and use the functions.
- Properly numpydoc and inline type hint your functions.

---

# Activity
Create and Use Your Own Python Module

** **

### Steps:

1. **Create a Python Module:**
    - In your working directory, create a new file called `my_module.py`.
    - Add two simple functions to the file:
      ```python
      # my_module.py

      def add(a, b):
          return a + b

      def subtract(a, b):
          return a - b
      ```

---

# Activity
Create and Use Your Own Python Module

** **

2. **Write a Script to Use Your Module:**
    - In the same directory, create another Python file called `use_module.py`.
    - Import your module and use its functions:
      ```python
      # use_module.py

      import my_module

      result1 = my_module.add(5, 3)
      result2 = my_module.subtract(10, 4)

      print(f"Addition: {result1}")      # Output: Addition: 8
      print(f"Subtraction: {result2}")   # Output: Subtraction: 6
      ```

--- 

# Activity
Create and Use Your Own Python Module

** **

3. **Run the Script:**
    - In the command line or terminal, run the `use_module.py` script:
      ```bash
      python use_module.py
      ```

