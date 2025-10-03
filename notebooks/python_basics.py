import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")

@app.cell

def _():
    """# Python Basics Overview

Welcome to a quick introduction to Python! In this notebook we’ll cover:
- Variables and data types
- Basic operators
- Control flow: if/else, loops
- Functions and modules
- A couple of interactive examples using marimo UI.
"""
    return

@app.cell

def _():
    """## Variables and Data Types

Python automatically infers the type of a variable when you assign a value to it.

```python
name = "Alice"          # str
age = 30                # int
height = 1.7            # float
is_student = False      # bool
```

You can check the type using `type()`.
```
print(type(name))
```
"""
    return

@app.cell

def _():
    """## Basic Operators

Python supports arithmetic, comparison, and logical operators.
```
# Arithmetic
sum_value = 3 + 5        # 8

# Comparison
print(5 > 2)            # True

# Logical
print(True and False)   # False
```
"""
    return

@app.cell

def _():
    """## Control Flow

Conditional statements:
```
if age > 18:
    print("Adult")
else:
    print("Minor")
```

Loops:
```
# For loop
for i in range(5):
    print(i)

# While loop
count = 0
while count < 3:
    print(count)
    count += 1
```
"""
    return

@app.cell

def _():
    """## Functions

Define reusable code blocks with `def`.
```
def greet(person):
    return f"Hello, {person}!"

print(greet("Bob"))
```
You can also use default arguments and keyword arguments.
```
def add(a, b=10):
    return a + b

print(add(5))   # 15
print(add(5, 20))  # 25
```
"""
    return

@app.cell

def _(mo):
    mo.md("## Quick Interactive Demo: Toggle a Boolean")
    return

@app.cell

def _(mo):
    toggle = mo.ui.toggle(label="Toggle", value=False)
    return toggle

@app.cell

def _(toggle):
    if toggle.value:
        toggle.md("The toggle is **ON**!")
    else:
        toggle.md("The toggle is **OFF**.")
    return

if __name__ == "__main__":
    app.run()
