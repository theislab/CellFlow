from contextlib import contextmanager
import time

__all__ = ["timer"]

@contextmanager
def timer(description: str, verbose: bool = True):
    """Context manager for timing operations with optional verbose output."""
    if verbose:
        start_time = time.time()
        print(f"{description}...")
    
    yield
    
    if verbose:
        end_time = time.time()
        print(f"{description} took {end_time - start_time:.2f} seconds.")