import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def claude_status(mo):
    mo.md("""
    **Claude connected** 🤝 — ready to pair on this notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
