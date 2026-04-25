# potato

## llvm2c

`llvm2c` converts LLVM IR (`.ll`) textual-format files to readable C source code.

### Requirements

* Python 3.8 or later (no third-party dependencies)

### Usage

```bash
# Convert a file and print C to stdout
python llvm2c.py input.ll

# Write C to a file
python llvm2c.py input.ll -o output.c

# Read LLVM IR from stdin
clang -emit-llvm -S -o - hello.c | python llvm2c.py -
```

### What is converted

| LLVM IR construct | C output |
|---|---|
| Integer types `i1`/`i8`/`i16`/`i32`/`i64` | `bool` / `int8_t` … `int64_t` |
| Float types `float`/`double` | `float` / `double` |
| Opaque pointer `ptr` | `void*` |
| Arrays `[N x T]` | `T[N]` |
| Global variables | top-level `T name = init;` |
| `define` functions | full C function with body |
| `declare` functions | C forward declaration |
| `alloca` | local pointer variable |
| `load` / `store` | dereference / assign |
| Binary arithmetic & bitwise ops | `+`, `-`, `*`, `/`, `%`, `&`, `\|`, `^`, `<<`, `>>` |
| Integer / float comparisons (`icmp`, `fcmp`) | `==`, `!=`, `<`, `>`, `<=`, `>=` |
| Type casts (`trunc`, `zext`, `sext`, `bitcast`, …) | C cast `(T)x` |
| `select` | ternary `cond ? a : b` |
| `phi` nodes | mutable variables with pre-branch assignments |
| `br` (conditional / unconditional) | `if (…) goto …; else goto …;` / `goto …;` |
| `switch` | `switch (val) { case … }` |
| `call` | C function call |
| `getelementptr` | pointer arithmetic |
| `unreachable` | `__builtin_unreachable()` |

### Example

Given `add.ll`:

```llvm
define i32 @add(i32 %a, i32 %b) {
  %result = add i32 %a, %b
  ret i32 %result
}
```

Running `python llvm2c.py add.ll` produces:

```c
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

int32_t add(int32_t a, int32_t b);

int32_t add(int32_t a, int32_t b) {
  entry:;
    int32_t result = a + b;
    return result;
}
```

### Running tests

```bash
pip install pytest
python -m pytest tests/
```