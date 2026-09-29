---
layout: center
---

# Practical Tools for Running Python Software
## How we actually build and run Python code

---

# Executing Python Software
Command Line

** **

Using `python script.py` or running the Python interpreter directly from the terminal.

<v-click>

- **Step 1: Navigate to your Working Directory.** 

    Use the `cd` command to navigate to the folder where you want to create and execute your Python script. 
    ```
    cd path/to/your/folder
    ```

</v-click>

---


# Executing Python Software
Command Line

** **

<v-click>

- **Step 3: Create a Simple Python Script.**  

    Use a text editor to create a new Python script called hello.py with the following content.
    ```
    # hbd.py
    print("Happy birthday, Kaitlynn!")
    ```

</v-click>
<v-click>

- **Step 4: Execute the Python Script.**  

    Run the script using the following command:
    ```
    python hbd.py
    ```

</v-click>

---

# Executing Python Software
Command Line

** **

<v-click>
  
  - **Advantages:**
    - Simple and quick for running standalone scripts.
    - Great for automation and batch processing.
    - Efficient for executing complete programs.

</v-click>
<v-click>
<br>

  - **Disadvantages:**
    - Limited debugging capabilities.
    - No interactivity once the script is running.
    - Less suitable for exploratory analysis or iterative development.

</v-click>

---

# Executing Python Software
Interactive Python

** **

**Open your *Prompt* and enter: `python`**

<v-click>

You will see something like:

```console
Python 3.13.13 | packaged by conda-forge | (main, Apr  8 2026, 01:56:49) [MSC v.1944 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

</v-click>
<v-click>

You can freely enter Python commands, and the session will remember what you enter:

```console
>>> x = 10
>>> print(x)
10
>>> x -= 5
>>> print(x)
5
```

Enter `quit()` to stop.

</v-click>

---

# Practical Tools for Running Python Software
Setting up Spyder

** **

<v-click>

**Command line is fine for quick testing, but we need something better for full projects.**

We need an ***Integrated Development Environment (IDE)***

</v-click>
<v-click>

This course uses ***Spyder*** because it is a clear, script-based Python environment that works well for beginners, students, and research workflows.

- Open Spyder from Anaconda Navigator or from the command line:
  ```bash
  spyder
  ```
- In Spyder, create a new file with **File > New File**.
- Write Python code in the editor, then run it with **F5** or the Run button.
- Results appear in the console, and variables can be inspected in the Variable Explorer.

</v-click>

---

# Practical Tools for Running Python Software
Creating and running a script in Spyder

** **

A Python script is a file containing code that runs from top to bottom.

```python
# hello.py
print("Hello, World!")
```

- Save the file as `hello.py`.
- Press **F5** in Spyder to run it.
- The output will appear in the IPython console.

---

# Practical Tools for Running Python Software
Why Spyder works well here

** **

<v-click>

- Beginner-friendly layout: editor, console, and variable explorer
- Good for script-based learning
- Easy to inspect variables while debugging
- Works well for scientific Python, data analysis, and plotting

</v-click>
<v-click>

The Spyder workflow

** **

1. Write code in the editor.
2. Save the file.
3. Run it with **F5**.
4. Check the output in the console.
5. Inspect variables in the Variable Explorer.

</v-click>

---
layout: center
---

# Activity
Get *Spyder* running

** **

**First thing's first: Let's use proper Environment Management. Make sure your conda env is active!**

```console
(base) C:\Users\conor>conda activate intro-to-python
```

** **

**If using *Anaconda Distribution*, it is included.**

**If using *Miniconda* or *Miniforge*, we need to install it...**

```console
(intro-to-python) C:\Users\conor> conda install spyder -y
```

** **

**Run *Spyder***
```console
(intro-to-python) C:\Users\conor> spyder
```

*Side Note:* Spyder is a <u>package</u>! Written in python, for python. We will get to packages later on.
