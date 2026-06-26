try:
    from .. import _hspice_read
except ImportError:
    _hspice_read = None


def hspice_read(filename, debug=0):
    """
    Read an HSPICE raw results file.
    
    Parameters:
    	filename: Path to the HSPICE raw file to read.
    	debug: Debug level passed to the reader.
    
    Returns:
    	The parsed HSPICE raw data.
    
    Raises:
    	ImportError: If the optional _hspice_read extension is unavailable.
    """
    if _hspice_read is None:
        raise ImportError(
            "HSpice read extension _hspice_read is not compiled/available."
        )
    return _hspice_read.hspice_read(filename, debug)

