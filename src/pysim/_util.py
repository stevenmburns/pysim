import matplotlib

# Backends with no display capability; calling show() on them is a no-op
# that emits a UserWarning. Useful for headless test runs.
_NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def is_interactive_backend():
    return matplotlib.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS


def save_or_show(plt, fn):
    if fn is not None:
        if fn != "/dev/null":
            plt.savefig(fn)
    elif is_interactive_backend():
        plt.show()

    plt.close()
