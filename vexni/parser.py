"""
Vexni Parser
Converts tokens into an Abstract Syntax Tree (AST)
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Union
from .lexer import Token, TokenType

# AST Node Classes
@dataclass
class ASTNode:
    """Base class for all AST nodes"""
    pass

@dataclass
class IndexOperation(ASTNode):
    target: ASTNode
    index: ASTNode


@dataclass
class Program(ASTNode):
    """Root node of the program"""
    functions: List['FunctionDeclaration']

@dataclass
class FunctionDeclaration(ASTNode):
    """Function definition node"""
    name: str
    parameters: List[tuple[str, str]]  # List of (name, type) pairs
    body: List[ASTNode]
    return_type: Optional[str] = None

@dataclass
class VariableDeclaration(ASTNode):
    """Variable declaration node"""
    name: str
    var_type: str
    initializer: Optional[ASTNode] = None

@dataclass
class Assignment(ASTNode):
    """Assignment node"""
    target: str
    value: ASTNode
    operator: str = '='  # For +=, -=, etc.

@dataclass
class BinaryOperation(ASTNode):
    """Binary operation node"""
    left: ASTNode
    operator: str
    right: ASTNode

@dataclass
class UnaryOperation(ASTNode):
    """Unary operation node"""
    operator: str
    operand: ASTNode

@dataclass
class IfStatement(ASTNode):
    """If statement node"""
    condition: ASTNode
    then_branch: List[ASTNode]
    elif_branches: List[tuple[ASTNode, List[ASTNode]]]
    else_branch: Optional[List[ASTNode]] = None

@dataclass
class ForLoop(ASTNode):
    """For loop node"""
    iterator: str
    iterable: ASTNode
    body: List[ASTNode]

@dataclass
class WhileLoop(ASTNode):
    """While loop node"""
    condition: ASTNode
    body: List[ASTNode]

@dataclass
class FunctionCall(ASTNode):
    """Function call node"""
    name: str
    arguments: List[ASTNode]

@dataclass
class CVFunctionCall(ASTNode):
    """Computer Vision function call node"""
    function: str
    arguments: List[ASTNode]

@dataclass
class Literal(ASTNode):
    """Literal value node"""
    value: Any
    literal_type: str

@dataclass
class Variable(ASTNode):
    """Variable reference node"""
    name: str

@dataclass
class Return(ASTNode):
    """Return statement node"""
    value: Optional[ASTNode] = None

class ParseError(Exception):
    """Custom exception for parsing errors"""
    pass

class VexniParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
        self.had_error = False
        self.cv_functions = {
            'detect_objects', 'classify_image', 'detect_faces',
            'extract_text', 'generate_image', 'load_image', 'save_image'
        }

    def parse(self) -> Program:
        """Parse the entire program"""
        try:
            functions = []
            while not self.is_at_end():
                functions.append(self.parse_function_declaration())
            return Program(functions)
        except ParseError as e:
            self.had_error = True
            raise e

    def parse_function_declaration(self) -> FunctionDeclaration:
        """Parse a function declaration"""
        self.consume(TokenType.FUNCTION, "Expected 'function' keyword")
        name = self.consume(TokenType.IDENTIFIER, "Expected function name").value

        # Parse parameters
        self.consume(TokenType.LPAREN, "Expected '(' after function name")
        parameters = []
        
        if not self.check(TokenType.RPAREN):
            while True:
                param_type = self.parse_type()
                param_name = self.consume(TokenType.IDENTIFIER, "Expected parameter name").value
                parameters.append((param_name, param_type))
                
                if not self.match(TokenType.COMMA):
                    break

        self.consume(TokenType.RPAREN, "Expected ')' after parameters")
        self.consume(TokenType.LBRACE, "Expected '{' before function body")
        
        body = self.parse_block()
        
        self.consume(TokenType.RBRACE, "Expected '}' after function body")
        
        return FunctionDeclaration(name, parameters, body)

    def parse_block(self) -> List[ASTNode]:
        """Parse a block of statements"""
        statements = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            statements.append(self.parse_statement())
        return statements

    def parse_statement(self) -> ASTNode:
        """Parse a single statement"""
        if self.match(TokenType.IF):
            return self.parse_if_statement()
        if self.match(TokenType.FOR):
            return self.parse_for_loop()
        if self.match(TokenType.WHILE):
            return self.parse_while_loop()
        if self.match(TokenType.RETURN):
            return self.parse_return_statement()
        if self.is_type_token():
            return self.parse_variable_declaration()
        
        # Expression statement
        expr = self.parse_expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after expression")
        return expr

    def parse_if_statement(self) -> IfStatement:
        """Parse an if statement"""
        self.consume(TokenType.LPAREN, "Expected '(' after 'if'")
        condition = self.parse_expression()
        self.consume(TokenType.RPAREN, "Expected ')' after condition")

        self.consume(TokenType.LBRACE, "Expected '{' before if body")
        then_branch = self.parse_block()
        self.consume(TokenType.RBRACE, "Expected '}' after if body")

        elif_branches = []
        while self.match(TokenType.ELIF):
            self.consume(TokenType.LPAREN, "Expected '(' after 'elif'")
            elif_condition = self.parse_expression()
            self.consume(TokenType.RPAREN, "Expected ')' after elif condition")
            
            self.consume(TokenType.LBRACE, "Expected '{' before elif body")
            elif_body = self.parse_block()
            self.consume(TokenType.RBRACE, "Expected '}' after elif body")
            
            elif_branches.append((elif_condition, elif_body))

        else_branch = None
        if self.match(TokenType.ELSE):
            self.consume(TokenType.LBRACE, "Expected '{' before else body")
            else_branch = self.parse_block()
            self.consume(TokenType.RBRACE, "Expected '}' after else body")

        return IfStatement(condition, then_branch, elif_branches, else_branch)

    def parse_for_loop(self) -> ForLoop:
        """Parse a for loop"""
        self.consume(TokenType.LPAREN, "Expected '(' after 'for'")
        iterator = self.consume(TokenType.IDENTIFIER, "Expected iterator variable").value
        
        self.consume(TokenType.IN, "Expected 'in' after iterator")
        iterable = self.parse_expression()
        
        self.consume(TokenType.RPAREN, "Expected ')' after for loop header")
        self.consume(TokenType.LBRACE, "Expected '{' before loop body")
        
        body = self.parse_block()
        
        self.consume(TokenType.RBRACE, "Expected '}' after loop body")

        return ForLoop(iterator, iterable, body)

    def parse_while_loop(self) -> WhileLoop:
        """Parse a while loop"""
        self.consume(TokenType.LPAREN, "Expected '(' after 'while'")
        condition = self.parse_expression()
        self.consume(TokenType.RPAREN, "Expected ')' after condition")

        self.consume(TokenType.LBRACE, "Expected '{' before loop body")
        body = self.parse_block()
        self.consume(TokenType.RBRACE, "Expected '}' after loop body")

        return WhileLoop(condition, body)

    def parse_return_statement(self) -> Return:
        """Parse a return statement"""
        value = None
        if not self.check(TokenType.SEMICOLON):
            value = self.parse_expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after return statement")
        return Return(value)

    def parse_variable_declaration(self) -> VariableDeclaration:
        """Parse a variable declaration"""
        try:
            # Get the type
            var_type = self.parse_type()
            
            # Get the variable name
            name = self.consume(TokenType.IDENTIFIER, "Expected variable name").value
            
            # Handle initialization if present
            initializer = None
            if self.match(TokenType.ASSIGN):
                initializer = self.parse_expression()
            
            # Require semicolon
            self.consume(TokenType.SEMICOLON, "Expected ';' after variable declaration")
            return VariableDeclaration(name, var_type, initializer)
        except Exception as e:
            print(f"Debug - Current token: {self.peek().type}, value: {self.peek().value}")
            raise e

    def parse_expression(self) -> ASTNode:
        """Parse an expression"""
        try:
            expr = self.parse_assignment()
            return expr
        except Exception as e:
            print(f"Debug - Error in expression parsing: {self.peek().type}")
            raise e

    def parse_function_call(self, name: str) -> Union[FunctionCall, CVFunctionCall]:
        """Parse a function call"""
        self.consume(TokenType.LPAREN, "Expected '(' after function name")
        arguments = []
        
        if not self.check(TokenType.RPAREN):
            while True:
                arguments.append(self.parse_expression())
                if not self.match(TokenType.COMMA):
                    break
        
        self.consume(TokenType.RPAREN, "Expected ')' after arguments")
        
        # Check if it's a CV function
        if name in self.cv_functions:
            return CVFunctionCall(name, arguments)
        return FunctionCall(name, arguments)

    def parse_assignment(self) -> ASTNode:
        """Parse an assignment expression"""
        expr = self.parse_logical_or()

        if self.match(TokenType.ASSIGN, TokenType.PLUS_ASSIGN, 
                     TokenType.MINUS_ASSIGN, TokenType.MULT_ASSIGN, 
                     TokenType.DIV_ASSIGN):
            operator = self.previous().value
            value = self.parse_assignment()
            
            if isinstance(expr, Variable):
                return Assignment(expr.name, value, operator)
                
            raise ParseError("Invalid assignment target")

        return expr

    def parse_logical_or(self) -> ASTNode:
        """Parse logical OR expressions"""
        expr = self.parse_logical_and()

        while self.match(TokenType.OR):
            operator = self.previous().value
            right = self.parse_logical_and()
            expr = BinaryOperation(expr, operator, right)

        return expr

    def parse_logical_and(self) -> ASTNode:
        """Parse logical AND expressions"""
        expr = self.parse_equality()

        while self.match(TokenType.AND):
            operator = self.previous().value
            right = self.parse_equality()
            expr = BinaryOperation(expr, operator, right)

        return expr

    def parse_equality(self) -> ASTNode:
        """Parse equality expressions"""
        expr = self.parse_comparison()

        while self.match(TokenType.EQUAL, TokenType.NOT_EQUAL):
            operator = self.previous().value
            right = self.parse_comparison()
            expr = BinaryOperation(expr, operator, right)

        return expr

    def parse_comparison(self) -> ASTNode:
        """Parse comparison expressions"""
        expr = self.parse_term()

        while self.match(TokenType.LESS, TokenType.GREATER, 
                        TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL):
            operator = self.previous().value
            right = self.parse_term()
            expr = BinaryOperation(expr, operator, right)

        return expr

    def parse_term(self) -> ASTNode:
        """Parse addition and subtraction"""
        expr = self.parse_factor()

        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.previous().value
            right = self.parse_factor()
            expr = BinaryOperation(expr, operator, right)

        return expr

    def parse_factor(self) -> ASTNode:
        """Parse multiplication and division"""
        expr = self.parse_unary()

        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
            operator = self.previous().value
            right = self.parse_unary()
            expr = BinaryOperation(expr, operator, right)

        return expr

    def parse_unary(self) -> ASTNode:
        """Parse unary expressions"""
        if self.match(TokenType.NOT, TokenType.MINUS):
            operator = self.previous().value
            right = self.parse_unary()
            return UnaryOperation(operator, right)

        return self.parse_primary()

    def parse_primary(self) -> ASTNode:
        """Parse primary expressions and handle indexing (postfix operations)"""
        try:
            if self.match(TokenType.BOOL):
                # Convert "true"/"false" string to boolean
                return Literal(self.previous().value.lower() == "true", "bool")

            if self.match(TokenType.INTEGER):
                return Literal(int(self.previous().value), "int")

            if self.match(TokenType.FLOAT):
                return Literal(float(self.previous().value), "float")

            if self.match(TokenType.STRING):
                return Literal(str(self.previous().value), "string")

            # Handle CV functions specifically
            if self.check_cv_function():
                token = self.advance()
                expr = self.parse_function_call(token.value)
            elif self.match(TokenType.IDENTIFIER):
                name = self.previous().value
                # Check if it's a function call
                if self.check(TokenType.LPAREN):
                    expr = self.parse_function_call(name)
                else:
                    expr = Variable(name)
            elif self.match(TokenType.LPAREN):
                # Parenthesized expression
                expr = self.parse_expression()
                self.consume(TokenType.RPAREN, "Expected ')' after expression")
            else:
                raise ParseError(f"Expected expression, got {self.peek().type}")

            # Handle indexing operations: variable[index], expr[index]
            while self.match(TokenType.LBRACKET):
                index_expr = self.parse_expression()
                self.consume(TokenType.RBRACKET, "Expected ']' after index expression")
                expr = IndexOperation(target=expr, index=index_expr)

            return expr

        except Exception as e:
            print(f"Debug - Error in primary expression: {self.peek().type}")
            raise e

    def check_cv_function(self) -> bool:
        """Check if current token is a CV function"""
        return self.peek().type in {
            TokenType.LOAD_IMAGE,
            TokenType.SAVE_IMAGE,
            TokenType.DETECT_OBJECTS,
            TokenType.CLASSIFY_IMAGE,
            TokenType.DETECT_FACES,
            TokenType.EXTRACT_TEXT,
            TokenType.GENERATE_IMAGE
        }

    def parse_type(self) -> str:
        """Parse a type declaration"""
        if not self.is_type_token():
            raise ParseError(f"Expected type, got {self.peek().type}")
        return self.advance().value

    def is_type_token(self) -> bool:
        """Check if current token is a type token"""
        return self.check(TokenType.INT_TYPE) or \
               self.check(TokenType.FLOAT_TYPE) or \
               self.check(TokenType.STRING_TYPE) or \
               self.check(TokenType.LIST_TYPE) or \
               self.check(TokenType.DICT_TYPE) or \
               self.check(TokenType.IMAGE_TYPE) or \
               self.check(TokenType.BOOL_TYPE)

    def match(self, *types: TokenType) -> bool:
        """Match current token against given types"""
        for token_type in types:
            if self.check(token_type):
                self.advance()
                return True
        return False

    def check(self, type: TokenType) -> bool:
        """Check if current token is of given type"""
        if self.is_at_end():
            return False
        return self.peek().type == type

    def advance(self) -> Token:
        """Advance to next token"""
        if not self.is_at_end():
            self.current += 1
        return self.previous()

    def is_at_end(self) -> bool:
        """Check if we've reached end of tokens"""
        return self.peek().type == TokenType.EOF

    def peek(self) -> Token:
        """Look at current token"""
        return self.tokens[self.current]

    def previous(self) -> Token:
        """Get previous token"""
        return self.tokens[self.current - 1]

    def consume(self, type: TokenType, message: str) -> Token:
        """Consume token of expected type or raise error"""
        if self.check(type):
            return self.advance()
        
        error_msg = f"{message} at line {self.peek().line}, column {self.peek().column}"
        raise ParseError(error_msg)

    def synchronize(self) -> None:
        """Recover from parse errors"""
        self.advance()

        while not self.is_at_end():
            if self.previous().type == TokenType.SEMICOLON:
                return

            if self.peek().type in {
                TokenType.FUNCTION,
                TokenType.IF,
                TokenType.FOR,
                TokenType.WHILE,
                TokenType.RETURN,
                TokenType.INT_TYPE,
                TokenType.FLOAT_TYPE,
                TokenType.STRING_TYPE,
                TokenType.LIST_TYPE,
                TokenType.DICT_TYPE,
                TokenType.IMAGE_TYPE,
                TokenType.BOOL_TYPE
            }:
                return

            self.advance()

    def report_error(self, token: Token, message: str) -> None:
        """Report a parsing error"""
        if token.type == TokenType.EOF:
            error_msg = f"Error at end of file: {message}"
        else:
            error_msg = f"Error at '{token.value}' line {token.line}, column {token.column}: {message}"
        
        self.had_error = True
        raise ParseError(error_msg)