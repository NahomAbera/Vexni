class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]

    def eat(self, token_type):
        if self.current_token[0] == token_type:
            self.pos += 1
            if self.pos < len(self.tokens):
                self.current_token = self.tokens[self.pos]
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token[0]}")

    def factor(self):
        if self.current_token[0] == 'NUMBER':
            val = self.current_token[1]
            self.eat('NUMBER')
            return int(val)
        elif self.current_token[0] == 'LPAREN':
            self.eat('LPAREN')
            result = self.expr()
            self.eat('RPAREN')
            return result
        else:
            raise SyntaxError("Invalid factor")

    def term(self):
        result = self.factor()
        while self.current_token[0] in ('MULT', 'DIV'):
            if self.current_token[0] == 'MULT':
                self.eat('MULT')
                result *= self.factor()
            elif self.current_token[0] == 'DIV':
                self.eat('DIV')
                result /= self.factor()
        return result

    def expr(self):
        result = self.term()
        while self.current_token[0] in ('PLUS', 'MINUS'):
            if self.current_token[0] == 'PLUS':
                self.eat('PLUS')
                result += self.term()
            elif self.current_token[0] == 'MINUS':
                self.eat('MINUS')
                result -= self.term()
        return result
