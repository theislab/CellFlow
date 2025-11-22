try:
    from scaleflow.external._scvi import CFJaxSCVI
except ImportError as e:
    raise ImportError(
        "scaleflow.external requires more dependencies. Please install via pip install 'scaleflow[external]'"
    ) from e
