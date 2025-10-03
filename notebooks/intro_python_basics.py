import marimo as mo

__generated_with = "0.13.0"
app = marimo.App(width="medium")

@app.cell
def _():
    mo.md("## Intro to Python Basics\nThis notebook demonstrates basic Python concepts such as variables, data types, control flow, loops, functions, and classes.")
    return

@app.cell
def _():
    # Variables and data types
    int_var = 10
    float_var = 3.14
    str_var = "Hello, World!"
    bool_var = True
    list_var = [1, 2, 3]
    tuple_var = (4, 5)
    dict_var = {"key": "value"}
    return int_var, float_var, str_var, bool_var, list_var, tuple_var, dict_var

@app.cell
def mo, int_var, float_var, str_var, bool_var, list_var, tuple_var, dict_var():
    # Display types
    types_info = {
        "int": type(int_var),
        "float": type(float_var),
        "string": type(str_var),
        "bool": type(bool_var),
        paced="list",;
    }
    mo.md(str(types_info))
    return types_info

@app.cell
def _():
    # Control flow
    if int_var > 5:
        result = "greater"
    else:
        result = "less or equal"
    return result

@app.cell
def mo, result():
    mo.md(f"Comparison of int_var > 5: {result}")
    return

@app.cell
def _():
    # For loop
    sum_numbers = 0
    for n in range(1, 6):  # 1 to 5
        sum_numbers += n
    return sum_numbers

@app.cell
def mo, sum_numbers():
    mo.md(f"Sum of numbers 1 to 5 is {sum_numbers}")
    return

@app.cell
def _():
    # Function definition
    def greet(name: str) -> str:
        return f"Hello, {name}!"
    greeting = greet("Alice")
    return greeting

@app.cell
def mo, greeting():
    mo.md(greeting)
    return

@app.cell
def _():
    # Simple class example
    class Counter:
        def __init__(self, start=0):
            self.value = start
        def increment(self):
            self.value += 1
            return self.value
    counter = Counter()
    counter.increment()
    counter.increment()
    return counter

@app.cell
def counter_value(counter):
    return counter.value

@app.cell
def mo, counter_value():
    mo.md(f"Counter value after two increments: {counter_value}")
    return