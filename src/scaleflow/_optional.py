class OptionalDependencyNotAvailable(ImportError):
    pass


def torch_required_msg() -> str:
    return (
        "Optional dependency 'torch' is required for this feature.\n"
        "Install it via: pip install torch  # or pip install 'scaleflow-tools[torch]'"
    )
