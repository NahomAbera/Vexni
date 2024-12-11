"""
Vexni - A Domain Specific Language for Computer Vision
"""

from .lexer import VexniLexer
from .parser import VexniParser
from .interpreter import VexniInterpreter
from .runner import VexniRunner

__version__ = "1.0.0"
__author__ = "Nahom Abera"
__email__ = "nahomtesfahun001@gmail.com"
__license__ = "MIT"

def run_vexni_file(file_path: str) -> None:
    """
    Execute a .vexni file
    
    Args:
        file_path (str): Path to the .vexni file
    """
    runner = VexniRunner()
    return runner.run_file(file_path)

def run_vexni_code(source_code: str) -> None:
    """
    Execute Vexni source code directly
    
    Args:
        source_code (str): Vexni source code as string
    """
    runner = VexniRunner()
    return runner.run_code(source_code)

__all__ = [
    'VexniLexer',
    'VexniParser',
    'VexniInterpreter',
    'VexniRunner',
    'run_vexni_file',
    'run_vexni_code',
    '__version__'
]