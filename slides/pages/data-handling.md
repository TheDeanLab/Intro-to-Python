---
layout: center
---

# Working with Data.
## Reading and Writing Data Across Operating Systems


---

# Working with Data
Basics of Folder and File Operations

** **

- **Basics of Folder and File Operations**
  - Python provides several modules to interact with the file system, making it easy to navigate folders, list files, and perform operations.
  - **Common Modules:**
    - `os`: Basic operating system functions for working with the file system.
    - `glob`: For pattern matching in filenames.
    - `pathlib`: A modern, object-oriented approach to handle file paths.

---

# Working with Data
Basics of Folder and File Operations

** **


- **Using `os` for Folder/File Operations**
  - **Change Directory:** 
    ```python
    import os
    os.chdir('/path/to/directory')
    ```
  - **List Files in Directory:**
    ```python
    files = os.listdir('/path/to/directory')
    print(files)
    ```
  - **Create a Directory:**
    ```python
    os.mkdir('new_folder')
    ```

`os` works across platforms and provides direct access to system-level commands.

---

# Working with Data
Basics of Folder and File Operations

** **

- **Using `glob` for Pattern Matching**
  - **Basic Pattern Matching:**
    - `glob` allows you to search for files matching a specific pattern, such as all `.txt` files in a folder.
    ```python
    import glob
    txt_files = glob.glob('*.txt')
    print(txt_files)
    ```
  - **Recursive Search (Python 3.5+):**
    ```python
    all_files = glob.glob('**/*.txt', recursive=True)
    print(all_files)
    ```

`glob` permits simple pattern-based searches for files within directories.


---

# 50% of time: File and folder manipulation

<v-click>

- **Real data is heterogeneous:** collected in different ways, different data formats
</v-click>

<v-click>

- **Every user is different:** different file naming, different folder stuctures (and different between experiments)
</v-click>

<v-click>

- **Operating systems are different:** different filepath convention
	- Windows: r"C:\Work\Projects\Intro-to-Python-2024\slides"
	- Windows: "C:\\\\Work\\\\Projects\\\\Intro-to-Python-2024\\\\slides"
	- UNIX: "/home/Work/Projects/Intro-to-Python-2024/slides"
</v-click>

<v-click>

- <ins>Most of the time</ins> is spent on locating the necessary files, reading and parsing them for the desired data in order to perform analysis:
	- **First 50%** is finding and reading the right files in the right folders into data structures (Built-in Python libraries)
	- **Second 50%** is writing code to extract the data in the right format to use for analysis (Day 1 + libraries)
</v-click>


---
layout: image-right
image: /images/glob_example.png
backgroundSize: 60%
---

# 50% of time: File and folder manipulation
`glob` library for file & folder search

- [`glob.glob`](https://docs.python.org/3/library/glob.html) enables finding files and folders matching a pattern. 
- pattern is given by [regular expression](https://www.geeksforgeeks.org/write-regular-expressions/#) (regex) e.g. `*` denotes wildcard  
<div class="flex flex-col items-center">
  <div>
    <img src="/images/glob_example_script.png" width="700" />
  </div>
</div>

- `recursive=True` searches also subfolders
- returns list of unordered file or folder paths 
- use `sorted()` or `numpy.sort()` to sort

---

# 50% of time: File and folder manipulation
`os` library for operating system (OS) operations

<v-click>

- Get current working directory (`os.getcwd`) and change to new working directory (`os.chdir`)
```python
import os
cwd = os.getcwd()
os.chdir('/path/to/new/directory')
```
</v-click>

<v-click>

- Create a path from strings (`os.path.join`)
```python
os.path.join('prefix/path', 'middle/path', 'suffix/path')
```
</v-click>

<v-click>

- Creating a directory (`os.mkdir`) and all intermediate directories (`os.mkdirs`)
```python
os.mkdir('/path/to/new/directory')
os.makedirs('/path/to/new/directory/with/intermediate/directories')
```
</v-click>


<v-click>

- Checking if path is a folder (`os.path.isdir`) or file (`os.path.isfile`)
```python
os.path.isdir('/path/to/directory')
os.path.isfile('/path/to/file')
```
</v-click>


---

# Working with Data
Basics of Folder and File Operations

** **

- **Introducing `pathlib`**
  - **Basic Path Creation:**
    - `pathlib` provides an object-oriented approach to handle paths and file operations, and it is cross-platform.
    ```python
    from pathlib import Path
    path = Path('/path/to/directory')
    ```

  - **Listing Files in a Directory:**
    ```python
    for file in path.iterdir():
        print(file)
    ```

`pathlib` simplifies path manipulation and handles cross-platform compatibility effortlessly.
  
---

# Working with Data
Basics of Folder and File Operations

** **

- **Pathlib Advanced Operations**
  - **Check if File or Directory Exists:**
    ```python
    if path.exists():
        print("Path exists.")
    ```

  - **Joining Paths and Accessing File Attributes:**
    ```python
    new_path = path / 'new_file.txt'
    print(new_path.name)         # Outputs 'new_file.txt'
    print(new_path.suffix)       # Outputs '.txt'
    print(new_path.parent)       # Outputs '/path/to/directory'
    ```

`pathlib` is a powerful and user-friendly module for file and folder operations in Python, especially for cross-platform projects.


---
layout: center
---

# Activity
Working with Files and Folders in Python


---

# Activity 1
Working with Files and Folders in Python
** **

1. **Using `os` for Folder/File Operations**
    - **Change Directory and Create a Folder:**
      - Open a Python script or interactive environment and run the following:
        ```python
        import os
        os.chdir('/path/to/your/working/directory')
        os.mkdir('activity_folder')
        ```
    - **List Files in the Current Directory:**
      - Inside the same directory, list all files:
        ```python
        files = os.listdir()
        print(files)
        ```

---

# Activity 2
Working with Files and Folders in Python
** **

2. **Using `glob` for Pattern Matching**
    - **Find All `.txt` Files in a Directory:**
      - Inside `activity_folder`, create a few `.txt` files and use `glob` to find them:
        ```python
        import glob
        txt_files = glob.glob('activity_folder/*.txt')
        print(txt_files)
        ```
    - **Recursive Pattern Matching (If using subdirectories):**
      - If you have nested folders, try recursive searching:
        ```python
        all_txt_files = glob.glob('activity_folder/**/*.txt', recursive=True)
        print(all_txt_files)
        ```

---

# Activity 3
Working with Files and Folders in Python
** **

3. **Using `pathlib` for Cross-Platform Path Operations**
    - **Create a Path Object and List Files in the Directory:**
      - Use `pathlib` to list files and access file properties:
        ```python
        from pathlib import Path
        path = Path('activity_folder')

        for file in path.iterdir():
            print(file.name)   # Print each file name
        ```
    - **Create a New File Using `pathlib`:**
      - Use `pathlib` to create a new file path and write to it:
        ```python
        new_file = path / 'example.txt'
        new_file.write_text("Hello, this is a test file.")
        ```

---

# Activity 4
Working with Files and Folders in Python
** **

4. **Challenge: Check File Types and Count Files**
    - **Count Files by Type:**
      - Using `pathlib`, count the number of `.txt` files in `activity_folder`:
        ```python
        txt_files_count = sum(1 for f in path.glob('*.txt'))
        print(f"Number of .txt files: {txt_files_count}")
        ```


---

# Working with Data
Introduction to File I/O (Input/Output)
** **

- **Overview:**
    - File handling allows Python programs to interact with files on the system, such as reading data, writing results, and appending new content.
    - Python provides built-in methods for file operations, making it easy to work with text and binary files.

- **Common File Operations:**
  - **Open:** Opens a file for reading or writing.
  - **Read:** Reads data from a file.
  - **Write:** Writes data to a file.
  - **Close:** Closes the file when done.

---

# Working with Data
Introduction to File I/O (Input/Output)
** **


- **File Access Modes**
  - Different modes allow for specific operations:
    - **Read (`r`):** Opens a file for reading. Fails if the file does not exist.
    - **Write (`w`):** Opens a file for writing. Creates a new file or overwrites an existing one.
    - **Append (`a`):** Opens a file for appending. Data is added at the end of the file without overwriting.
    - **Read & Write (`r+`):** Opens a file for both reading and writing.

- **Examples of Opening Files:**
  ```python
  file = open("example.txt", "r")  # Open for reading
  file = open("example.txt", "w")  # Open for writing (overwrites)
  file = open("example.txt", "a")  # Open for appending


---

# Working with Data
Introduction to File I/O (Input/Output)
** **


- **Reading Files**
  - **Read Entire File (`read`):** Reads the entire content as a single string.
    ```python
    with open("example.txt", "r") as file:
        content = file.read()
        print(content)
    ```

  - **Read Lines (`readlines`):** Reads all lines and returns them as a list.
    ```python
    with open("example.txt", "r") as file:
        lines = file.readlines()
        print(lines)
    ```

  - **Read Line-by-Line (`for` loop):**
    ```python
    with open("example.txt", "r") as file:
        for line in file:
            print(line.strip())  # .strip() removes newline characters
    ```

---

# Working with Data
Introduction to File I/O (Input/Output)
** **

- **Writing Files**
  - **Write Text (`write`):** Writes a string to a file.
    ```python
    with open("example.txt", "w") as file:
        file.write("Hello, World!")
    ```

  - **Write Multiple Lines (`writelines`):** Writes a list of strings to a file.
    ```python
    lines = ["Hello, World!\n", "This is a test.\n"]
    with open("example.txt", "w") as file:
        file.writelines(lines)
    ```

---

# Working with Data
Introduction to File I/O (Input/Output)
** **

- **Appending Text to Files**
  - **Append Mode (`a`):** Adds new content to the end of the file without overwriting existing content.
    ```python
    with open("example.txt", "a") as file:
        file.write("Additional line of text.\n")
    ```


---

# Working with Data
Introduction to File I/O (Input/Output)
** **

- **Best Practice: Using Context Managers**
  - Using a `with` statement automatically handles file closing after operations, even if an error occurs.
  - Example:
    ```python
    with open("example.txt", "r") as file:
        content = file.read()  # No need to explicitly close
    ```

- **Advantages of Context Managers:**
  - Ensures file resources are properly released.
  - Reduces risk of file corruption or data loss.
  - Cleaner code with no need for `file.close()`.

---
layout: center
---

# Activity
Introduction to File I/O (Input/Output)

---

# Activity 5: Reading and Counting Lines
Introduction to File I/O (Input/Output)
** **

**Objective:** Read a text file and count the number of lines.


**Instructions:**
1. Create a text file called `sample.txt` with some sample text (at least 5 lines).
2. Write a Python script to open the file, read it line by line, and count the total number of lines.
3. Print the line count.

<v-click>

**Solution:**
```python
with open("sample.txt", "r") as file:
    line_count = sum(1 for line in file)
print(f"Total lines: {line_count}")
```
</v-click>


---

# Activity 6: Writing and Appending to a File
Introduction to File I/O (Input/Output)
** **

**Objective:** Write data to a new file, then append additional data.

**Instructions:**

Create a script that writes a list of apples, bananas, and carrots to a file called `shopping_list.txt` using `writelines()`. Open the same file in append mode and add tomatoes and potatoes to the file. Finally, print the contents of the file to verify the data was added.

<v-click>

**Solution:**

```python
items = ["apples\n", "bananas\n", "carrots\n"]
with open("shopping_list.txt", "w") as file:
    file.writelines(items)

with open("shopping_list.txt", "a") as file:
    file.write("tomatoes\n")
    file.write("potatoes\n")

with open("shopping_list.txt", "r") as file:
    print(file.read())
```

</v-click>

---

# Activity 7: Filtering Lines Based on Keywords
Introduction to File I/O (Input/Output)
** **

**Objective:** Read a file and print only lines that contain a specific keyword.

**Instructions:**

1. Create a text file called `log.txt` with at least 10 lines of sample text. Make sure some lines include the word `"ERROR"`.
   
2. Write a script that opens `log.txt` and reads each line.
   
3. Print only the lines that contain the word `"ERROR"`.

<v-click>

**Solution:**
```python
with open("log.txt", "r") as file:
    for line in file:
        if "ERROR" in line:
            print(line.strip())
```

</v-click>

---

# Activity 8: Basic Statistics from a File
Introduction to File I/O (Input/Output)
** **

**Objective:** Read numerical data from a file and calculate the average.

**Instructions:**

1. Create a file named `numbers.txt` with one integer per line (at least 5 numbers).
   
2. Write a Python script to open the file, read the numbers, and calculate the average.
   
3. Print the average value.

<v-click>

**Solution:**

```python
with open("numbers.txt", "r") as file:
    numbers = [int(line.strip()) for line in file]

average = sum(numbers) / len(numbers)
print(f"Average: {average}")
```

</v-click>

---

# Activity 9: Using Context Managers for Multiple Files
Introduction to File I/O (Input/Output)
** **


**Objective:** Read data from one file and write it to another using a context manager.

**Instructions:**

1. Create a file called `source.txt` with some text data.
   
2. Write a script that reads the contents of `source.txt` and writes it to a new file named `destination.txt`.
   
3. Use a context manager to handle both files.

<v-click>

**Solution:**
```python
with open("source.txt", "r") as source, open("destination.txt", "w") as dest:
    content = source.read()
    dest.write(content)
```

</v-click>


---
layout: center
---

# Working with Data
Introduction to Error Handling in Python


---

# Working with Data
Introduction to Error Handling in Python
** **

- **What is Error Handling?**
  - Error handling allows Python programs to gracefully handle unexpected issues.
  - Common file and folder errors include:
    - **FileNotFoundError:** Raised when attempting to access a file that doesn’t exist.
    - **PermissionError:** Raised when the program lacks permission to access a file or folder.

- **Using `try`/`except` Statements**
  - `try`/`except` blocks allow you to catch and handle errors without stopping the entire program.

---

# Working with Data
Introduction to Error Handling in Python
** **

- **Basic Syntax of `try`/`except`**
  - Place the code that may cause an error inside the `try` block.
  - Use an `except` block to define how to handle specific errors.
  - Example:
    ```python
    try:
        with open("nonexistent_file.txt", "r") as file:
            content = file.read()
    except FileNotFoundError:
        print("Error: File not found.")
    ```

---

# Working with Data
Introduction to Error Handling in Python
** **

- **Using Multiple `except` Blocks for Specific Errors**
  - You can handle different types of errors separately by using multiple `except` blocks.
  - Example:
    ```python
    try:
        with open("file.txt", "r") as file:
            content = file.read()
    except FileNotFoundError:
        print("Error: The file does not exist.")
    except PermissionError:
        print("Error: You do not have permission to access this file.")
    except (IsADirectoryError, NotADirectoryError) as e:
        print("You can even combine multiple exceptions and get the error response...")
        print(f"The error is: {e})
    ```

---

# Working with Data
Introduction to Error Handling in Python
** **

- **Best Practices for Error Handling in File Operations**
  - **Use Context Managers with `try`/`except`:** The `with` statement ensures files are properly closed, even if an error occurs.
  - **Be Specific with Error Types:** Handle specific exceptions to avoid masking unexpected issues.
  - **Log Errors:** Print or log error messages to understand what went wrong and for easier debugging.

- **Example Combining Context Managers and Error Handling:**
    ```python
    try:
        with open("data.txt", "r") as file:
            content = file.read()
    except FileNotFoundError:
        print("Error: File not found.")
    except PermissionError:
        print("Error: Permission denied.")
    ```

---

# Activity 10: Handling Missing Files
Introduction to Error Handling in Python
** **


**Objective:** Use `try`/`except` to handle errors when a file is missing.

**Instructions:**
1. Write a script that tries to open a file called `data.txt` for reading.
2. If the file does not exist, catch the `FileNotFoundError` and print a message saying `"File not found. Please check the file path."`

<v-click>

**Solution:**
```python
try:
    with open("data.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("File not found. Please check the file path.")
```

</v-click>


---
layout: center
---

# Working with Data
Introduction to Generators


---

# Working with Data
Introduction to Generators

** **

- **Introduction to Generators**
  - Generators are a special type of function that yield values one at a time using the `yield` keyword instead of `return`.
  - Unlike functions that return all values at once, generators produce values on demand, making them memory-efficient for processing large data streams.
  
- **Benefits of Generators:**
  - **Memory Efficiency:** Generators yield one item at a time, reducing memory use.
  - **Lazy Evaluation:** Generators only produce values when needed, which is ideal for large or infinite data streams.
  - **Simplifies Code:** Generators simplify writing iterators, allowing for more readable and concise code.

---

# Working with Data
Introduction to Generators

** **

- **Creating a Basic Generator with `yield`**
  - Generators are defined similarly to functions but use `yield` to produce a sequence of values.
  - Example:
    ```python
    def simple_generator():
        yield "First"
        yield "Second"
        yield "Third"
        
    for value in simple_generator():
        print(value)
    # Output:
    # First
    # Second
    # Third
    ```

---

# Working with Data
Introduction to Generators

** **

- **Use Case: Processing Large Data Streams**
  - Generators are commonly used for processing data in chunks, such as reading large files line by line:
    ```python
    def read_large_file(file_path):
        with open(file_path, "r") as file:
            for line in file:
                yield line.strip()

    for line in read_large_file("large_file.txt"):
        print(line)  # Processes one line at a time, conserving memory
    ```

## Key Takeaway:
**Generators** provide memory-efficient data processing for large datasets by yielding items one at a time.


---
layout: center
---

# Working with Data
Introduction to Iterators

---

# Working with Data
Introduction to Iterators

** **

- **Introduction to `itertools` for Advanced Iteration**
  - `itertools` is a Python module providing functions for creating efficient iterators.
  - These tools simplify complex data processing tasks, allowing you to work with infinite sequences, chain multiple iterables, filter data, and more.

- **Useful Functions in `itertools`:**
  - **`count`**: Generates an infinite sequence of numbers.
  - **`cycle`**: Repeats elements of an iterable indefinitely.
  - **`chain`**: Combines multiple iterables into one.
  - **`islice`**: Slices an iterable, allowing you to control start, stop, and step.

---

# Working with Data
Introduction to Iterators

** **

- **Examples of `itertools` Functions**
  - **Counting Sequence with `count`:**
    ```python
    from itertools import count

    for i in count(10, 2):
        if i > 20:
            break
        print(i)
    # Output: 10, 12, 14, 16, 18, 20
    ```

---

# Working with Data
Introduction to Iterators

  - **Cycling Through a Sequence with `cycle`:**
    ```python
    from itertools import cycle

    counter = 0
    for item in cycle(["A", "B", "C"]):
        print(item)
        counter += 1
        if counter == 6:
            break
    # Output: A, B, C, A, B, C
    ```

---

# Working with Data
Introduction to Iterators

** **

- **Chaining with `itertools`**
  - **Chaining Iterables with `chain`:**
    ```python
    from itertools import chain

    combined = chain([1, 2, 3], ["a", "b", "c"])
    for item in combined:
        print(item)
    # Output: 1, 2, 3, a, b, c
    ```

---

# Working with Data
Introduction to Iterators

** **
- **Slicing with `itertools`**
  - **Slicing with `islice`:**
    ```python
    from itertools import islice

    for number in islice(count(0), 5, 10):
        print(number)
    # Output: 5, 6, 7, 8, 9
    ```

## Key Takeaway:
**`itertools`** offers advanced iteration tools that streamline tasks like counting, cycling, and chaining, enhancing the efficiency of data processing workflows.

---

# Activity 11: Processing Data Streams with Generators and `itertools`

Introduction to Iterators and Generators
** **

**Objective:** Use a generator to process a simulated data stream and combine it with `itertools` functions to manipulate the data efficiently.

**Instructions:**

1. Define a generator function called `data_stream` that yields integers from 1 to 100.
2. Use `itertools.islice` to take only a subset of values from the generator (e.g., 10 values starting from the 5th value).
3. Create a cyclic sequence of labels (`"A"`, `"B"`, `"C"`) using `itertools.cycle`. Pair each value from the sliced data with a label from the cycle, so that each item from `data_stream` gets a label.
4. The output should be pairs like `(5, "A")`, `(6, "B")`, `(7, "C")`, continuing with `(8, "A")` and so on, for 10 values.

---

# Activity 11: Processing Data Streams with Generators and `itertools`
Introduction to Iterators and Generators

** **

**Solution:**
```python
from itertools import islice
# Step 1: Define a generator for a data stream
def data_stream():
    for i in range(1, 101):
        yield i
# Step 2: Use itertools.islice to take a subset of values
stream_slice = islice(data_stream(), 4, 14)  # Starts at 5th item and takes 10 items
# Step 3: Use itertools.cycle to create cyclic labels and pair with data
labels = cycle(["A", "B", "C"])
paired_data = zip(stream_slice, labels)
# Step 4: Print the resulting pairs
for item in paired_data:
    print(item)
# Expected Output:
# (5, 'A')
# (6, 'B')
# (7, 'C')
# (8, 'A')
# (9, 'B')
# ...
```

---
layout: center
---

# Working with Data
Basic Data Operations (Filtering, Sorting, ...)

---

# Working with Data
Basic Data Operations (Filtering, Sorting, ...)

** **

- **Introduction to Basic Data Operations**
  - Python’s core data structures (lists, dictionaries, tuples) are powerful tools for data manipulation.
  - Common operations:
    - **Filtering:** Select specific data points that meet a condition.
    - **Sorting:** Arrange data in a particular order.
    - **Basic Statistics:** Perform simple calculations like sums, averages, or counts.

- **Core Data Structures:**
  - **Lists:** Ordered and mutable, useful for storing sequences.
  - **Dictionaries:** Key-value pairs, fast lookup and retrieval.
  - **Tuples:** Ordered and immutable, often used for fixed data collections.

---

# Working with Data
Basic Data Operations (Filtering, Sorting, ...)

** **

- **Filtering Data in Python**
  - Filtering is commonly done with list comprehensions or the `filter()` function.
  - **Using List Comprehension:**
    ```python
    numbers = [1, 2, 3, 4, 5, 6]
    evens = [n for n in numbers if n % 2 == 0]
    print(evens)  # Output: [2, 4, 6]
    ```

  - **Using `filter()` Function:**
    ```python
    def is_even(n):
        return n % 2 == 0

    evens = list(filter(is_even, numbers))
    print(evens)  # Output: [2, 4, 6]
    ```

---

# Working with Data
Basic Data Operations (Filtering, Sorting, ...)
** **


- **Filtering Dictionaries:**
  - Filter based on keys or values in a dictionary.
    ```python
    scores = {"Alice": 85, "Bob": 90, "Charlie": 75}
    passed = {k: v for k, v in scores.items() if v >= 80}
    print(passed)  # Output: {'Alice': 85, 'Bob': 90}
    ```

- **Sorting Data in Python**
  - **Sorting Lists with `sorted()` and `sort()`:**
    - `sorted()` returns a new sorted list, while `sort()` sorts in place.
    ```python
    numbers = [3, 1, 4, 1, 5]
    sorted_numbers = sorted(numbers)
    print(sorted_numbers)  # Output: [1, 1, 3, 4, 5]
    ```


---

# Working with Data
Basic Data Operations (Filtering, Sorting, ...)
** **


  - **Sorting with Custom Key Functions:**
    - Sort using a key function, such as by length or by specific value.
    ```python
    words = ["apple", "banana", "kiwi"]
    sorted_words = sorted(words, key=len)
    print(sorted_words)  # Output: ['kiwi', 'apple', 'banana']
    ```

- **Sorting Dictionaries by Values:**
  - Sort dictionary items by values using `sorted()`:
    ```python
    scores = {"Alice": 85, "Bob": 90, "Charlie": 75}
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    print(sorted_scores)  # Output: {'Bob': 90, 'Alice': 85, 'Charlie': 75}
    ```

---

# Working with Data
Basic Data Operations (Filtering, Sorting, ...)
** **


- **Basic Statistics in Python**
  - **Calculating the Sum and Average:**
    - Use `sum()` and `len()` to calculate the total and average.
    ```python
    numbers = [3, 7, 2, 9]
    total = sum(numbers)
    average = total / len(numbers)
    print(f"Total: {total}, Average: {average}")
    # Output: Total: 21, Average: 5.25
    ```

  - **Finding Maximum and Minimum Values:**
    - Use `max()` and `min()` to find the largest and smallest values.
    ```python
    numbers = [3, 7, 2, 9]
    print(f"Max: {max(numbers)}, Min: {min(numbers)}")
    # Output: Max: 9, Min: 2
    ```

---

# Working with Data
Basic Data Operations (Filtering, Sorting, ...)

** **

- **Using the `statistics` Module for Basic Statistics**
  - Python’s `statistics` module provides additional functions for descriptive statistics.
  - **Mean, Median, and Mode:**
    ```python
    import statistics

    data = [1, 2, 3, 4, 5, 5]
    print(f"Mean: {statistics.mean(data)}")       # Output: Mean: 3.33
    print(f"Median: {statistics.median(data)}")   # Output: Median: 3.5
    print(f"Mode: {statistics.mode(data)}")       # Output: Mode: 5
    ```

## Key Takeaway:
- **Filtering, sorting, and calculating basic statistics** are essential data operations that can be performed efficiently with core Python data structures.
- Python’s built-in functions and the `statistics` module make it easy to perform these operations with minimal code.

---