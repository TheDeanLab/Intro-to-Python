---
layout: center
---

# Executing Python Software
Command Line

** **



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

# Executing Python Software
Command Line

** **

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

# Executing Python Software
Command Line

** **

  - **Advantages:**
    - Simple and quick for running standalone scripts.
    - Great for automation and batch processing.
    - Efficient for executing complete programs.
  - **Disadvantages:**
    - Limited debugging capabilities.
    - No interactivity once the script is running.
    - Less suitable for exploratory analysis or iterative development.
---

# Creating a Python Environment
Why do we need environments?

** **

<v-clicks>

- Imagine you’re working on two projects:
  - One needs **Python 3.10** and **NumPy 1.20**
  - Another needs **Python 3.12** and **NumPy 2.0**
- If everything installs in the same place… they **clash**!
- Your computer won’t know which version to use.

</v-clicks>

---

# Creating a Python Environment
What is an environment?

** **

<v-clicks>

- An **environment** is like a **mini workspace** inside your computer.
- Each environment has its **own Python** and its **own packages**.
- You can switch between them anytime — like having multiple “toolboxes”.

</v-clicks>


---

# Creating a Python Environment
Why use environments?

** **

<v-clicks>

- Keep different projects **separate**  
- **Avoid breaking** old code when you install something new  
- Make it **easy to share** your setup with others  
  
</v-clicks>


---

# Creating a Python Environment
Steps to create a basic environment

** **


- Open your terminal or command prompt and type the following

<v-click>
```bash
conda create -n myenv python=3.12 -y
```
</v-click>


- Then activate you environment:

<v-click>
```bash
conda activate myenv
```
</v-click>

<v-click>
Now you have a clean space with just Python installed!
You can add packages later (e.g., conda install numpy).
</v-click>


