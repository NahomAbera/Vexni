import re

tokens = [
    ('NUMBER', r'\d+(\.\d*)?'),   # Integer or decimal number
    ('PLUS', r'\+'),              # Addition
    ('MINUS', r'-'),              # Subtraction
    ('MULT', r'\*'),              # Multiplication
    ('DIV', r'/'),                # Division
    ('LPAREN', r'\('),            # Left Parenthesis
    ('RPAREN', r'\)'),            # Right Parenthesis
    ('WHITESPACE', r'\s+'),       # Whitespace
]

def lex(characters):
    pos = 0
    while pos < len(characters):
        match = None
        for token_type, regex in tokens:
            regex = re.compile(regex)
            match = regex.match(characters, pos)
            if match:
                if token_type != 'WHITESPACE': 
                    yield (token_type, match.group(0))
                pos = match.end(0)
                break
        if not match:
            raise SyntaxError(f'Illegal character: {characters[pos]}')
