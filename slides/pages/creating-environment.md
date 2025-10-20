---
layout: center
---

# Creating a Python Environment

---

# Creating a Python Environment
Why do we need environments?

** **

<v-clicks>

- Imagine you’re working on two projects:
  - One needs **Python 3.10** and **NumPy 1.20**
  - Another needs **Python 3.12** and **NumPy 2.0**
- If everything installs in the same place… 🧨 they **clash**!
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
✅ Now you have a clean space with just Python installed!
You can add packages later (e.g., conda install numpy).
</v-click>


