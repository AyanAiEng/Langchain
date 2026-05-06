
# 🧠 Python Decorators (Basics → Advanced)

## ❓ What is a decorator?

A **decorator** is a function that:
- Takes another function
- Adds extra behavior
- Returns a modified function

👉 In simple words:
Decorator = function wrapper

---

## ⚡ What does “wrap a function” mean?

Wrapping means:
- Putting your function inside another function
- Running extra code before/after it

Example structure:

```
wrapper()
   → before code
   → original function
   → after code
```

---

## 🧪 Basic Example

```python
def decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper
```

---

## ⚡ Using @ syntax

```python
@decorator
def hello():
    print("Hello")
```

Same as:

```python
hello = decorator(hello)
```

---

## 🧠 What enhancement does a decorator do?

A decorator can add:
- Logging
- Timing
- Authentication
- Validation
- Modification of input/output

---

## 🔥 Real-world use cases

- @login_required
- @timer
- @cache
- @tool (LangChain)

---

## 🧠 Why decorators are powerful

They let you:
- Extend behavior
- Without changing original function

---

## 🚀 Advanced idea

A decorator:
- Takes function
- Creates wrapper
- Returns enhanced function object

---

## ⚡ Final summary

A decorator is a function that wraps another function and adds extra behavior before or after execution without modifying original code.
