"""
Compatibility shim.

The canonical tools live in `tools/ds_tools.py`. This file re-exports them so
older imports (`import ds_tools`) keep working.
"""

from tools.ds_tools import (  # noqa: F401
    ALL_TOOLS,
    inspect_dataframe,
    install_package,
    list_directory,
    read_file,
    run_python,
    write_file,
)
