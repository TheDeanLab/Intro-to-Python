---
layout: center
---

# Functions and Modules 

---

# Introduction to Functions in Python
Starting Simple.

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

- **Benefits of Using Functions:**
  - **Code Reusability:**
    - Functions allow you to write code once and reuse it wherever needed, avoiding repetition.
  
  - **Modularity:**
    - By breaking down complex problems into smaller functions, you make your code more organized and easier to understand.

  - **Maintainability:**
    - Functions allow you to easily update or debug parts of your code without affecting the rest of the program.

---

# Introduction to Functions in Python

  - **Readability:**
    - Well-named functions make the code more readable by summarizing what specific sections of the code do.

  - **Avoiding Global Variables:**
    - Functions help avoid the misuse of global variables by encapsulating variables locally within the function.



---

# Introduction to Functions in Python
Functions with return values


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

- **What are `*args` in Functions?**
  - `*args` allows you to pass a variable number of positional arguments to a function.
  - The `*` operator collects all extra positional arguments passed to the function and bundles them into a tuple, making it easy to iterate over them or process multiple inputs.
  - This makes functions more flexible, allowing them to accept varying numbers of arguments without needing to be rewritten.

- **Example of a Function Using `*args`:**
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

- **What are `**kwargs` in Functions?**
  - `**kwargs` allows you to pass a variable number of keyword arguments to a function.
  - The `**` operator collects these arguments into a dictionary, allowing you to handle named arguments dynamically.
  - This makes functions more flexible, enabling them to accept a varying number of keyword arguments without needing predefined parameters.

- **Example of a Function Using `**kwargs`:**
    ```python
    def print_info(**kwargs):
        for key, value in kwargs.items():
            print(f"{key}: {value}")
    
    print_info(name="Alice", age=30)
    # Output:
    # name: Alice
    # age: 30
    ```

