from lexer import lex
from parser import Parser

def interpret(expression):
    tokens = list(lex(expression))
    parser = Parser(tokens)
    result = parser.expr()
    return result

if __name__ == '__main__':
    while True:
        try:
            expression = input('Enter expression: ')
            result = interpret(expression)
            print(f'Result: {result}')
        except Exception as e:
            print(f'Error: {e}')
