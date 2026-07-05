# Gotchas

## Private variables are cell-scoped

Variables with a `_` prefix are **private to the cell that defines them** in
marimo. They cannot be referenced from other cells — you'll get a `NameError`.

This matters when building notebooks programmatically. A common mistake:

```python
# Cell A
_df = pd.DataFrame(results)   # _df is private to this cell

# Cell B — FAILS
mo.ui.table(_df)               # NameError: name '_df' is not defined
```

**Fix:** Either merge both into one cell, or use a non-private name (`df`).

## Redefining a public name across cells

Each public name has one owning cell. Defining it again in another cell fails
with `Multiply-defined names`. This is easy to hit when building a notebook
incrementally — a second cell reassigns `df`, `results`, `data`, etc.

```python
# Cell A
df = pd.read_csv("data.csv")

# Cell B — FAILS: df already defined in Cell A
df = df.dropna()               # Multiply-defined names: df
```

**Fix — pick one:**

- **Edit the owning cell** if the step belongs there (`ctx.edit_cell`).
- **Use a new name** when later cells need the result (`clean = df.dropna()`).
- **Use a private `_` name** for a throwaway intermediate (`_clean = df.dropna()`).

`ctx.graph.cells[cid].defs` shows what a cell already owns.

## Duplicate public imports across cells

The same single-definition rule applies to imports: a public name (like `pd`)
can only be defined in one cell. If two cells both `import pandas as pd`, you
get a `Multiply-defined names` error at validation.

**Fix:** Use a `_` prefix on the second import (`import pandas as _pd`) or
consolidate imports into a shared cell.

## `inspect.getsource()` on methods is indented

`inspect.getsource()` on a class method preserves the original indentation.
Passing this to `ast.parse()` fails with `IndentationError`.

```python
# FAILS
src = inspect.getsource(SomeClass.some_method)
tree = ast.parse(src)  # IndentationError: unexpected indent

# FIX
import textwrap
src = textwrap.dedent(inspect.getsource(SomeClass.some_method))
tree = ast.parse(src)
```

## Cached module availability

Some libraries cache optional-dependency availability at import time. Installing
a package mid-session via `ctx.packages.add()` won't update those caches.
The user may need to restart the kernel — but try known workarounds first.

### Local packages pushed into a molab sandbox

If a local package's `.py` files are written into a molab sandbox after the
kernel already `import`ed it once (e.g. via `sync-molab-notebook`'s
`push-package.py`), the running kernel keeps using the old, cached module
objects — updated files on disk don't retroactively change what's already
in `sys.modules`. This is the same caching class of problem as the
polars/pyarrow case below, but attempting a `importlib.reload()` across a
multi-module package is not a safe workaround here: stale class objects
(e.g. an old dataclass) can make `isinstance` checks fail against instances
created before the reload, producing confusing errors far from the actual
cause.

**Fix:** restart the kernel, or delete and re-create the cell(s) that
`import` the package, before re-running dependent cells. Don't try to patch
around it with `reload()`.

### Polars + pyarrow

`df.to_pandas()` fails with `ModuleNotFoundError: pa.Table requires 'pyarrow'`.

**Workaround** — if this error occurs after installing pyarrow mid-session,
run the following via `execute-code` (scratchpad), NOT in a cell. The patch
mutates the cached module object in the running kernel, so it doesn't need to
persist in the notebook.

```python
import pyarrow as _pa
import polars.dataframe.frame as _frame_mod
_frame_mod.pa = _pa
```

Then re-run the failing cell.
