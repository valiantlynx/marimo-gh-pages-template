import marimo as mo

__generated_with = "0.13.6"
app = mo.App(width="medium")

@app.cell

def _intro_md():
    mo.md(
        """
# Introduction to Python Basics

This notebook walks through Python core concepts in an interactive way. Each cell demonstrates a building block—from variables and types to control flow and functions.
"""
    )
    return

@app.cell

def _variables_and_types():
    a = 10           # integer
    b = 2.5          # float
    c = "Hello"     # string
    d = True         # boolean
    return a, b, c, d

@app.cell

def _operators():
    a, b = 10, 2.5
    sum_val = a + b
    diff_val = a - b
    prod_val = a * b
    div_val = a / b
    comp_val = a > b
    return sum_val, diff_val, prod_val, div_val, comp_val

@app.cell

def _conditionals():
    a = 10
    if a > 5:
        result = "a is greater than 5"
    else:
        result = "a is 5 or less"
    return result

@app.cell

def _loops():
    fruits = ["apple", "banana", "cherry"]
    for f in fruits:
        print(f"Found fruit: {f}")
    return fruits

@app.cell

def _functions():
    def greet(person):
        return f"Hello, {person}!"
    return greet

@app.cell

def _basic_io():
    message = "This is a simple print statement."
    print(message)
    return message

if __name__ == "__main__":
    app.run()
