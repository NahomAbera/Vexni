"""
Vexni Runner
Main entry point for executing Vexni programs
"""

import os
import sys
import argparse
from typing import Optional, Union, Dict, Any
from pathlib import Path

from .lexer import VexniLexer
from .parser import VexniParser
from .interpreter import VexniInterpreter

class VexniRunner:
    def __init__(self):
        """Initialize the Vexni runner"""
        self.interpreter = VexniInterpreter()
        self.debug_mode = False

    def run_file(self, file_path: Union[str, Path], debug: bool = False) -> Optional[Any]:
        """
        Execute a Vexni program from a file
        
        Args:
            file_path: Path to the .vexni file
            debug: Enable debug output
            
        Returns:
            Optional[Any]: Result of program execution
        """
        self.debug_mode = debug
        file_path = Path(file_path)

        # Verify file extension
        if file_path.suffix != '.vexni':
            raise ValueError("File must have .vexni extension")

        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Read source code
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            return self.run_code(source_code, str(file_path))

        except Exception as e:
            self._handle_error(e, file_path)
            if self.debug_mode:
                raise
            return None

    def run_code(self, source_code: str, source_name: str = "<string>") -> Optional[Any]:
        """
        Execute Vexni source code directly
        
        Args:
            source_code: Vexni source code as string
            source_name: Name for error reporting
            
        Returns:
            Optional[Any]: Result of program execution
        """
        try:
            # Tokenize
            if self.debug_mode:
                print(f"[DEBUG] Tokenizing source code...")
            lexer = VexniLexer(source_code)
            tokens = lexer.tokenize()

            if self.debug_mode:
                print(f"[DEBUG] Tokens generated:")
                for token in tokens:
                    print(f"  {token}")

            # Parse
            if self.debug_mode:
                print(f"[DEBUG] Parsing tokens...")
            parser = VexniParser(tokens)
            ast = parser.parse()

            if self.debug_mode:
                print(f"[DEBUG] AST generated:")
                self._print_ast(ast)

            # Interpret
            if self.debug_mode:
                print(f"[DEBUG] Executing program...")
            result = self.interpreter.interpret(ast)

            if self.debug_mode:
                print(f"[DEBUG] Execution completed. Result: {result}")

            return result

        except Exception as e:
            self._handle_error(e, source_name)
            if self.debug_mode:
                raise
            return None

    def _handle_error(self, error: Exception, source: Union[str, Path]) -> None:
        """Handle and format error messages"""
        error_type = type(error).__name__
        source_name = str(source)
        
        if self.debug_mode:
            print(f"\n[ERROR] {error_type} in {source_name}:", file=sys.stderr)
            print(f"  {str(error)}", file=sys.stderr)
            print("\nTraceback:", file=sys.stderr)
            import traceback
            traceback.print_exc()
        else:
            print(f"\nError in {source_name}:", file=sys.stderr)
            print(f"  {str(error)}", file=sys.stderr)

    def _print_ast(self, node, level: int = 0) -> None:
        """Print AST structure for debugging"""
        indent = "  " * level
        if hasattr(node, '__dict__'):
            print(f"{indent}{node.__class__.__name__}:")
            for key, value in node.__dict__.items():
                if isinstance(value, (list, tuple)):
                    print(f"{indent}  {key}:")
                    for item in value:
                        self._print_ast(item, level + 2)
                else:
                    print(f"{indent}  {key}: {value}")
        else:
            print(f"{indent}{node}")

def create_cli() -> argparse.ArgumentParser:
    """Create command-line interface"""
    parser = argparse.ArgumentParser(
        description="Vexni - Computer Vision Domain Specific Language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vexni script.vexni              # Run a Vexni script
  vexni script.vexni --debug      # Run with debug output
  vexni --version                 # Show version
        """
    )

    parser.add_argument(
        'file',
        nargs='?',
        help='Path to the .vexni file to execute'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'Vexni 1.0.0'
    )

    return parser

def main() -> None:
    """Main entry point for CLI"""
    parser = create_cli()
    args = parser.parse_args()

    if not args.file:
        parser.print_help()
        return

    try:
        runner = VexniRunner()
        runner.run_file(args.file, debug=args.debug)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()