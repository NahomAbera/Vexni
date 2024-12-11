"""
Vexni Lexical Analyzer (Lexer)
Converts source code into tokens for the parser.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List

class TokenType(Enum):
    # Keywords
    FUNCTION = auto()
    IF = auto()
    ELIF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    IN = auto()
    RETURN = auto()
    
    # Types
    INT_TYPE = auto()
    FLOAT_TYPE = auto()
    STRING_TYPE = auto()
    LIST_TYPE = auto()
    DICT_TYPE = auto()
    IMAGE_TYPE = auto()
    BOOL_TYPE = auto()
    
    # CV Keywords
    DETECT_OBJECTS = auto()
    CLASSIFY_IMAGE = auto()
    DETECT_FACES = auto()
    EXTRACT_TEXT = auto()
    GENERATE_IMAGE = auto()
    LOAD_IMAGE = auto()
    SAVE_IMAGE = auto()
    
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOL = auto()
    
    # Operators
    PLUS = auto()          # +
    MINUS = auto()         # -
    MULTIPLY = auto()      # *
    DIVIDE = auto()        # /
    MODULO = auto()        # %
    ASSIGN = auto()        # =
    PLUS_ASSIGN = auto()   # +=
    MINUS_ASSIGN = auto()  # -=
    MULT_ASSIGN = auto()   # *=
    DIV_ASSIGN = auto()    # /=
    
    # Comparison
    EQUAL = auto()         # ==
    NOT_EQUAL = auto()     # !=
    LESS = auto()          # <
    GREATER = auto()       # >
    LESS_EQUAL = auto()    # <=
    GREATER_EQUAL = auto() # >=
    
    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Delimiters
    LPAREN = auto()    # (
    RPAREN = auto()    # )
    LBRACE = auto()    # {
    RBRACE = auto()    # }
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    SEMICOLON = auto() # ;
    COMMA = auto()     # ,
    DOT = auto()       # .
    
    # Other
    IDENTIFIER = auto()
    EOF = auto()

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class VexniLexer:
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.tokens = []
        self.current = 0
        self.line = 1
        self.column = 1
        
        # Define keywords including CV functions
        self.keywords = {
            'function': TokenType.FUNCTION,
            'if': TokenType.IF,
            'elif': TokenType.ELIF,
            'else': TokenType.ELSE,
            'for': TokenType.FOR,
            'while': TokenType.WHILE,
            'in': TokenType.IN,
            'return': TokenType.RETURN,
            'int': TokenType.INT_TYPE,
            'float': TokenType.FLOAT_TYPE,
            'string': TokenType.STRING_TYPE,
            'list': TokenType.LIST_TYPE,
            'dict': TokenType.DICT_TYPE,
            'image': TokenType.IMAGE_TYPE,
            'bool': TokenType.BOOL_TYPE,
            'true': TokenType.BOOL,
            'false': TokenType.BOOL,
            'and': TokenType.AND,
            'or': TokenType.OR,
            'not': TokenType.NOT,
            # CV Keywords
            'detect_objects': TokenType.DETECT_OBJECTS,
            'classify_image': TokenType.CLASSIFY_IMAGE,
            'detect_faces': TokenType.DETECT_FACES,
            'extract_text': TokenType.EXTRACT_TEXT,
            'generate_image': TokenType.GENERATE_IMAGE,
            'load_image': TokenType.LOAD_IMAGE,
            'save_image': TokenType.SAVE_IMAGE
        }

    def tokenize(self) -> List[Token]:
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        
        self.tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return self.tokens

    def scan_token(self):
        c = self.advance()
        
        if c.isspace():
            if c == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            return
            
        if c.isdigit():
            self.number()
            return
            
        if c.isalpha() or c == '_':
            self.identifier()
            return
            
        if c == '"':
            self.string()
            return
            
        # Handle operators and punctuation
        simple_tokens = {
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            ',': TokenType.COMMA,
            '.': TokenType.DOT,
            ';': TokenType.SEMICOLON,
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.MULTIPLY,
            '/': TokenType.DIVIDE,
            '%': TokenType.MODULO
        }
        
        if c in simple_tokens:
            self.add_token(simple_tokens[c])
            return
            
        # Handle two-character operators
        if c == '=':
            self.add_token(TokenType.EQUAL if self.match('=') else TokenType.ASSIGN)
        elif c == '!':
            self.add_token(TokenType.NOT_EQUAL if self.match('=') else TokenType.NOT)
        elif c == '<':
            self.add_token(TokenType.LESS_EQUAL if self.match('=') else TokenType.LESS)
        elif c == '>':
            self.add_token(TokenType.GREATER_EQUAL if self.match('=') else TokenType.GREATER)
        else:
            raise SyntaxError(f"Unexpected character '{c}' at line {self.line}, column {self.column}")

    def identifier(self):
        while self.peek().isalnum() or self.peek() == '_':
            self.advance()

        text = self.source_code[self.start:self.current]
        token_type = self.keywords.get(text, TokenType.IDENTIFIER)
        self.add_token(token_type)

    def number(self):
        while self.peek().isdigit():
            self.advance()

        # Look for decimal part
        if self.peek() == '.' and self.peek_next().isdigit():
            self.advance()  # Consume the dot
            while self.peek().isdigit():
                self.advance()
            self.add_token(TokenType.FLOAT)
        else:
            self.add_token(TokenType.INTEGER)

    def string(self):
    # Advance until the closing quote or EOF
        while self.peek() != '"' and not self.is_at_end():
            if self.peek() == '\n':
                self.line += 1
            self.advance()

        if self.is_at_end():
            raise SyntaxError(f"Unterminated string at line {self.line}")

        self.advance()  # Consume the closing quote

        # Extract the string content without the quotes.
        # If self.start was at the opening quote, and self.current is now at the char after closing quote,
        # the actual string content is between self.start+1 and self.current-1.
        string_value = self.source_code[self.start+1:self.current-1]

        self.tokens.append(Token(TokenType.STRING, string_value, self.line, self.column))
        self.column += (self.current - self.start)


    def match(self, expected: str) -> bool:
        if self.is_at_end() or self.source_code[self.current] != expected:
            return False
        self.current += 1
        return True

    def advance(self) -> str:
        self.current += 1
        return self.source_code[self.current - 1]

    def peek(self) -> str:
        if self.is_at_end():
            return '\0'
        return self.source_code[self.current]

    def peek_next(self) -> str:
        if self.current + 1 >= len(self.source_code):
            return '\0'
        return self.source_code[self.current + 1]

    def is_at_end(self) -> bool:
        return self.current >= len(self.source_code)

    def add_token(self, type: TokenType):
        text = self.source_code[self.start:self.current]
        self.tokens.append(Token(type, text, self.line, self.column))
        self.column += self.current - self.start