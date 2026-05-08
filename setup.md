# Setup

## Prerequisites

- Python 3.12+
- A C++23 compiler (MSVC 2022, GCC 13+, or Clang 16+)
- CMake 3.20+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Steps

### 1. Install and Build

```bash
uv sync
```

### 2. Rebuild after changes

```
uv sync --reinstall
```

This compiles the C++ Grizzlar library and the pybind11 extension (`_grizzlars`) and installs `grizzlars` into the virtual environment.

### 3. Run the demo

```bash
uv run python main.py
```

---

## Troubleshooting

**CMake not found**
Install CMake from [cmake.org](https://cmake.org/download/) and ensure it is on your `PATH`.

**MSVC compiler errors (`/bigobj`, C++23)**
Open the project in a Visual Studio 2022 Developer Command Prompt, or install the "Desktop development with C++" workload via the Visual Studio Installer.

**Rebuild after changing C++ code**
