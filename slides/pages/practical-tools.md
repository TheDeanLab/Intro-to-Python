---
layout: center
---

# Practical Tools for Running Python Software
## Executing Python Software 


---

# Practical Tools for Running Python Software

**Command Line Execution** 

Using `python script.py` or running the Python interpreter directly from the terminal.

- **Step 1: Open the Command Line.**
    - Windows: Open Command Prompt (search for `cmd`or `Command Prompt`).
    - macOS/Linux: Open Terminal.

- **Step 2: Navigate to your Working Directory.** 

    Use the `cd` command to navigate to the folder where you want to create and execute your Python script. 
    ```
    cd path/to/your/folder
    ```

---

# Practical Tools for Running Python Software

**Command Line Execution** 

- **Step 3: Create a Simple Python Script.**  

    Use a text editor to create a new Python script called hello.py with the following content.
    ```
    # hello.py
    print("Hello, World!")
    ```

- **Step 4: Execute the Python Script.**  

    Run the script using the following command:
    ```
    python hello.py
    ```

---

# Practical Tools for Running Python Software

**Command Line Executation**

  - **Advantages:**
    - Simple and quick for running standalone scripts.
    - Great for automation and batch processing.
    - Efficient for executing complete programs.
  - **Disadvantages:**
    - Limited debugging capabilities.
    - No interactivity once the script is running.
    - Less suitable for exploratory analysis or iterative development.


---

# Practical Tools for Running Python Software

**Interactive Mode**

Entering the Python interpreter by typing `python` in the terminal and running code line-by-line interactively.

- **Step 1: Open the Command Line.**
    - Windows: Open Command Prompt (search for `cmd` or `Command Prompt`).
    - macOS/Linux: Open Terminal.

- **Step 2: Start the Python Interpreter.**  

    Enter the following command to start the interactive mode:
    ```
    python
    ```

    - Once in the interactive mode, your terminal will output information on the `Python` installation.
        ```
        Last login: Thu Oct 10 08:44:41 on ttys000
        (base) S155475@SW567797 slides % python
        Python 3.9.12 (main, Apr  5 2022, 01:52:34) 
        [Clang 12.0.0 ] :: Anaconda, Inc. on darwin
        Type "help", "copyright", "credits" or "license" for more information.
        ```

---

# Practical Tools for Running Python Software

**Interactive Mode**

Entering the Python interpreter by typing `python` in the terminal and running code line-by-line interactively.


- **Step 3: Test a Simple Python Command.** 

    Once the interpreter starts, you can run Python code line-by-line. For example:
    ```
    >>> print("Hello, World!")
    ```

- **Step 4: Define a Simple Variable and Perform a Calculation.**  

    Enter commands directly into the interpreter:
    ```
    >>> x = 5
    >>> y = 10
    >>> result = x + y
    >>> print(result)
    ```

- **Step 5: Exit the Python Interpreter.**

    To exit the interactive mode, type:
    ```
    >>> exit()
    ```

---

# Practical Tools for Running Python Software

**Interactive Mode**

  - **Advantages:**
    - Immediate feedback; great for testing small snippets of code.
    - Excellent for learning and experimentation.
    - No need to save or create files for quick code execution.
  - **Disadvantages:**
    - Not ideal for running large programs.
    - Requires manually saving work if you want to retain results.
    - Code can't easily be reused across different projects.


---

# Practical Tools for Running Python Software

**JupyterLab Notebooks**

A web-based interactive development environment for writing and executing Python code.

- **Step 1: If necessasry, install JupyterLab with pip.**
    ```
    pip install jupyterlab
    ```

- **Step 2: Launch JupyterLab.**  

    Open the command line and run the following command to start JupyterLab:
    ```
    jupyter-lab
    ```
    - JupyterLab will launch in the present working directory of your terminal.

- **Step 3: Create a New Notebook.**  

    Once JupyterLab is open in your browser, click on the "Python 3" option under the "Notebook" section to create a new Python notebook.


---

# Practical Tools for Running Python Software

**JupyterLab Notebooks**

- **Step 4: Run Code in Cells.** 

    In a Jupyter notebook, code is written in cells. Type the following in the first cell:
    ```
    print("Hello, World!")
    ```
    - To execute the code, press `Shift + Enter` or click the "Run" button.

- **Step 5: Use Markdown for Documentation.**  

    You can also create markdown cells for documentation. Switch a cell to markdown mode by selecting ***Markdown*** from the dropdown and typing text for documentation.

---

# Practical Tools for Running Python Software

**JupyterLab Notebooks**

  - **Advantages:**
    - Interactive environment perfect for data science, research, and documentation.
    - Supports code, markdown, and rich media in a single interface.
    - Excellent for visualization and step-by-step execution.
  - **Disadvantages:**
    - More resource-intensive compared to command-line execution.
    - Not ideal for large, standalone programs.
    - Can be cumbersome for version control and collaboration.


---

# Practical Tools for Running Python Software

**Spyder IDE**

An integrated development environment (IDE) specifically tailored for scientific computing with Python.

- **Step 1: Install Spyder (if not already installed).**

    You can install Spyder using pip:
    ```
    pip install spyder
    ```

- **Step 2: Launch Spyder.**

    Open the command line and run the following command to start Spyder:
    ```
    spyder
    ```

---

# Practical Tools for Running Python Software

**Spyder IDE**

An integrated development environment (IDE) specifically tailored for scientific computing with Python.


- **Step 3: Create or Open a Python Script.**
    
    Once Spyder is open, you can create a new Python script by going to **File > New File** or open an existing one using **File > Open**.
    Type the following in the script editor:
    ```
    print("Hello, World!")
    ```

- **Step 4: Execute the Python Script.**
    
    To run the script, press `F5` or click the "Run" button in the toolbar. The output will appear in the "Console" panel at the bottom.

- **Step 5: Explore Spyder's Features.**
    
    Spyder offers many useful features, such as variable explorer (view and edit variables), integrated help, and an interactive IPython console.  

---

# Practical Tools for Running Python Software

**Spyder IDE**

- **Advantages:**
  - **Integrated Development Environment:** Includes an interactive Python console, variable explorer, and script editor in a single interface.
  - **Built-in Visualization Tools:** Direct integration with popular libraries like Matplotlib, allowing real-time plotting and visualization.
  - **Interactive Debugging:** Provides robust debugging tools, including breakpoints and stepping through code to help with troubleshooting.
- **Disadvantages:**
  - **Resource-Intensive:** Requires more memory and CPU compared to simpler text editors or the command line.
  - **Not Ideal for Large Projects:** While great for small to medium scripts, Spyder might not be the best choice for managing large-scale software projects.
  - **Less Customizable:** Compared to IDEs like VS Code or PyCharm, Spyder is less customizable.


---
layout: center
---

# Practical Tools for Running Python Software
## One more thing... 

---

# Understanding `if __name__ == "__main__"`

- **What is `__name__`?**
  - In Python, `__name__` is a special built-in variable that represents the name of the current module.
  - When a Python file is run directly, `__name__` is set to `"__main__"`. However, when a file is imported as a module, `__name__` is set to the module's name instead.

- **Why use `if __name__ == "__main__":`?**
  - This statement ensures that a block of code is only executed when the script is run directly, not when it’s imported as a module in another script.
  - It is commonly used to encapsulate the "entry point" of a Python script.

---

# Understanding `if __name__ == "__main__"`

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


**Key Benefit:** 
- It allows for reusable code by preventing specific parts of a script from running when the script is imported as a module elsewhere.
