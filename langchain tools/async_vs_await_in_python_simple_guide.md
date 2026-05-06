# ⚡ Async vs Await in Python (Simple Guide)

---

## 🧠 Q1: What is `async`?

👉 `async` means: **This function is asynchronous and can pause without blocking the program**.

```python
async def get_user():
    print("Fetching user")
```

### ✔️ Simple meaning:
> This function is allowed to use `await` and can run in non-blocking way.

---

## ⏳ Q2: What is `await`?

👉 `await` means: **Pause this task here, but let other tasks run**.

```python
import asyncio

async def get_user():
    user = await asyncio.sleep(2)
    return user
```

### ✔️ Simple meaning:
> Wait for something to finish, but don’t freeze the whole program.

---

## 🔥 Key Relationship

| Keyword | Meaning |
|--------|--------|
| `async` | Makes function capable of pausing |
| `await` | Actually pauses the function |

---

## 🧠 Simple Analogy

### 🚶 Sync (normal code)
- You wait in line until your turn comes

### ⚡ Async/await
- You take a token → sit → do other things → get called later

---

## ⚡ Flow Example

```python
async def main():
    user = await fetch_user()
    print(user)
```

### What happens:

1. `async` → function becomes non-blocking
2. `await` → pauses only this step
3. Other tasks can run
4. Result comes back → function continues

---

## 🚀 Why we use it in industry

- 🌐 API calls (fast servers)
- 🗄️ Database queries
- 🤖 AI / LangChain tools
- 📡 Network requests
- 💬 Chat apps

👉 Because all these involve waiting

---

## 🧠 One-line memory

- `async` = “I can wait”
- `await` = “I will wait here, but let others run”

---

## ⚡ Final Summary

👉 `async` allows a function to run in non-blocking mode  
👉 `await` pauses only that function without blocking the whole program
