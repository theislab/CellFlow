from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

__all__ = ["AnnDataLocation"]

class AnnDataLocation:
    """
    An object that stores a sequence of access operations (attributes and keys)
    and can be called on an AnnData object to execute them.
    """
    def __init__(self, path=None):
        # The path is a list of tuples, e.g., [('getattr', 'obsm'), ('getitem', 's')]
        self._path = path if path is not None else []

    def __getattr__(self, name):
        """
        Handles attribute access, like .obs or .X.
        It returns a new AnnDataLocation with the attribute access added to the path.
        """
        if name.startswith('__') and name.endswith('__'):
            # Avoid interfering with special methods
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
        new_path = self._path + [('getattr', name)]
        return AnnDataLocation(new_path)

    def __getitem__(self, key):
        """
        Handles item access, like ['my_key'].
        It returns a new AnnDataLocation with the item access added to the path.
        """
        new_path = self._path + [('getitem', key)]
        return AnnDataLocation(new_path)

    def __call__(self, adata: 'AnnData'):
        """
        Executes the stored path of operations on the provided AnnData object.
        """
        target = adata
        try:
            for op_type, op_arg in self._path:
                if op_type == 'getattr':
                    target = getattr(target, op_arg)
                elif op_type == 'getitem':
                    target = target[op_arg]
            return target
        except (AttributeError, KeyError) as e:
            raise type(e)(f"Failed to resolve location {self!r} on the AnnData object. Reason: {e}") from e

    def __repr__(self):
        """Provides a user-friendly string representation of the stored path."""
        representation = "AnnDataAccessor()"
        for op_type, op_arg in self._path:
            if op_type == 'getattr':
                representation += f'.{op_arg}'
            elif op_type == 'getitem':
                # Use repr() to correctly handle string keys with quotes
                representation += f'[{repr(op_arg)}]'
        return f"<AnnDataLocation: {representation}>"
