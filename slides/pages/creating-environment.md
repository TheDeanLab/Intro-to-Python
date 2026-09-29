# How to use Anaconda?

** **

<v-click>

***Anaconda Distribution*** comes with a GUI: ***Navigator*** which can create environments, <br> but we don't recommend that for this course.

</v-click>
<v-click>

- **We recommend to use command line in your Interpreter:**
    - ***Windows:*** Open **Anaconda Prompt** <br>
      Search: `Anaconda Prompt` (or `Miniforge Prompt` if Miniforge)
    - ***macOS/Linux:*** Open Terminal. *Much easier!*

</v-click>

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

<v-click>

- Open your **Prompt** and type the following

```bash
conda create -n myenv python=3.11 -y
```
</v-click>
<br>

<v-click>
- Then activate you environment:

```bash
conda activate myenv
```
</v-click>
<br>

<v-click>
Now you have a clean space with just Python installed!
You can add packages later (e.g., conda install numpy).
</v-click>
<br>

---
layout: center
---

# Activity
Let's create an environment for this Course!

** **

**Open *Anaconda Prompt* (or Terminal, etc):**

Create the environment:
```console
(base) C:\Users\conor> conda create -n intro-to-python python=3.11 -y
```
Activate the environment:
```console
(base) C:\Users\conor> conda activate intro-to-python
```
Did it work?
```console
(intro-to-python) C:\Users\conor> python --version
Python 3.11.16
``` 

