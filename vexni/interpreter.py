"""
Vexni Interpreter
Executes the Abstract Syntax Tree (AST) and handles runtime operations
"""

import cv2
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import pytesseract
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from .parser import (
    ASTNode, Program, FunctionDeclaration, VariableDeclaration,
    Assignment, BinaryOperation, UnaryOperation, IfStatement,
    ForLoop, WhileLoop, FunctionCall, CVFunctionCall, Literal,
    Variable, Return, IndexOperation
)

class VexniRuntimeError(Exception):
    """Custom exception for runtime errors"""
    pass

@dataclass
class Environment:
    """Stores variables and functions in current scope"""
    variables: Dict[str, Any]
    functions: Dict[str, FunctionDeclaration]
    parent: Optional['Environment'] = None

    def define(self, name: str, value: Any) -> None:
        """Define a new variable in current scope"""
        self.variables[name] = value

    def assign(self, name: str, value: Any) -> None:
        """Assign value to existing variable"""
        if name in self.variables:
            self.variables[name] = value
        elif self.parent:
            self.parent.assign(name, value)
        else:
            raise VexniRuntimeError(f"Undefined variable '{name}'")

    def get(self, name: str) -> Any:
        """Get value of variable"""
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.get(name)
        raise VexniRuntimeError(f"Undefined variable '{name}'")

class BuiltinFunction:
    """Represents a built-in function"""
    def __init__(self, name: str, func):
        self.name = name
        self.func = func

    def call(self, args: List[Any]) -> Any:
        return self.func(*args)

class VexniInterpreter:
    def __init__(self):
        self.global_env = Environment({}, {})
        self.current_env = self.global_env
        self._setup_builtins()
        self._setup_cv_models()

    def _setup_builtins(self):
        """Set up built-in functions"""
        self.builtins = {
            'print': BuiltinFunction('print', self._builtin_print),
            'len': BuiltinFunction('len', self._builtin_len),
            'write_file': BuiltinFunction('write_file', self._builtin_write_file),
            'read_file': BuiltinFunction('read_file', self._builtin_read_file),
        }

    def _setup_cv_models(self):
        """Initialize computer vision models"""
        # Object Detection
        self.object_detector = YOLO('yolov8n.pt')
        
        # Image Classification
        self.classifier = pipeline("image-classification")
        
        # Face Detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # OCR Setup
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        
        # Image Generation
        self.image_generator = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0"
        )

    # Built-in function implementations
    def _builtin_print(self, *args):
        """Built-in print function"""
        print(*[str(arg) for arg in args])

    def _builtin_len(self, obj):
        """Built-in len function"""
        return len(obj)

    def _builtin_write_file(self, filename: str, content: str):
        """Built-in write_file function"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

    def _builtin_read_file(self, filename: str) -> str:
        """Built-in read_file function"""
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()

    def interpret(self, program: Program) -> None:
        """Execute the program"""
        # Register all functions first
        for func in program.functions:
            self.global_env.functions[func.name] = func
        
        # Look for main function
        if 'main' in self.global_env.functions:
            main_func = self.global_env.functions['main']
            return self._execute_function_call(main_func, [])
        else:
            raise VexniRuntimeError("No main function found")

    def execute(self, node: ASTNode) -> Any:
        """Execute an AST node"""
        method_name = f'_execute_{node.__class__.__name__}'
        method = getattr(self, method_name, self._execute_unknown)
        return method(node)

    def _execute_unknown(self, node: ASTNode) -> None:
        """Handle unknown node types"""
        raise VexniRuntimeError(f"Unknown node type: {node.__class__.__name__}")

    def _execute_FunctionDeclaration(self, node: FunctionDeclaration) -> None:
        """Execute function declaration"""
        self.current_env.functions[node.name] = node

    def _execute_VariableDeclaration(self, node: VariableDeclaration) -> None:
        """Execute variable declaration"""
        value = None
        if node.initializer:
            value = self.execute(node.initializer)
        self.current_env.define(node.name, value)

    def _execute_Assignment(self, node: Assignment) -> Any:
        """Execute assignment"""
        value = self.execute(node.value)
        self.current_env.assign(node.target, value)
        return value

    def _execute_BinaryOperation(self, node: BinaryOperation) -> Any:
        """Execute binary operation"""
        left = self.execute(node.left)
        right = self.execute(node.right)

        operators = {
            '+': lambda: left + right,
            '-': lambda: left - right,
            '*': lambda: left * right,
            '/': lambda: left / right,
            '%': lambda: left % right,
            '==': lambda: left == right,
            '!=': lambda: left != right,
            '<': lambda: left < right,
            '>': lambda: left > right,
            '<=': lambda: left <= right,
            '>=': lambda: left >= right,
            'and': lambda: left and right,
            'or': lambda: left or right,
        }

        if node.operator not in operators:
            raise VexniRuntimeError(f"Unknown operator: {node.operator}")

        return operators[node.operator]()

    def _execute_UnaryOperation(self, node: UnaryOperation) -> Any:
        """Execute unary operation"""
        operand = self.execute(node.operand)
        
        if node.operator == 'not':
            return not operand
        elif node.operator == '-':
            return -operand
        
        raise VexniRuntimeError(f"Unknown unary operator: {node.operator}")

    def _execute_IfStatement(self, node: IfStatement) -> None:
        """Execute if statement"""
        if self.execute(node.condition):
            for stmt in node.then_branch:
                self.execute(stmt)
        else:
            # Check elif branches
            for condition, body in node.elif_branches:
                if self.execute(condition):
                    for stmt in body:
                        self.execute(stmt)
                    return
            
            # Execute else branch if it exists
            if node.else_branch:
                for stmt in node.else_branch:
                    self.execute(stmt)

    def _execute_ForLoop(self, node: ForLoop) -> None:
        """Execute for loop"""
        iterable = self.execute(node.iterable)
        
        for item in iterable:
            loop_env = Environment({}, {}, self.current_env)
            loop_env.define(node.iterator, item)
            
            prev_env = self.current_env
            self.current_env = loop_env
            
            for stmt in node.body:
                self.execute(stmt)
            
            self.current_env = prev_env

    def _execute_WhileLoop(self, node: WhileLoop) -> None:
        """Execute while loop"""
        while self.execute(node.condition):
            for stmt in node.body:
                self.execute(stmt)

    def _execute_FunctionCall(self, node: FunctionCall) -> Any:
        """Execute function call"""
        # Check for built-in functions first
        if node.name in self.builtins:
            args = [self.execute(arg) for arg in node.arguments]
            return self.builtins[node.name].call(args)

        # Check for user-defined functions
        function = self.global_env.functions.get(node.name)
        if not function:
            raise VexniRuntimeError(f"Undefined function: {node.name}")

        args = [self.execute(arg) for arg in node.arguments]
        return self._execute_function_call(function, args)

    def _execute_CVFunctionCall(self, node: CVFunctionCall) -> Any:
        """Execute computer vision function call"""
        args = [self.execute(arg) for arg in node.arguments]
        
        cv_functions = {
            'load_image': self._cv_load_image,
            'save_image': self._cv_save_image,
            'detect_objects': self._cv_detect_objects,
            'classify_image': self._cv_classify_image,
            'detect_faces': self._cv_detect_faces,
            'extract_text': self._cv_extract_text,
            'generate_image': self._cv_generate_image,
            'draw_rectangle': self._cv_draw_rectangle
        }
        
        if node.function not in cv_functions:
            raise VexniRuntimeError(f"Unknown CV function: {node.function}")
            
        return cv_functions[node.function](*args)

    # Computer Vision function implementations
    def _cv_load_image(self, path: str):
        """Load an image from file"""
        return cv2.imread(path)

    def _cv_save_image(self, image, path: str):
        """Save an image to file"""
        cv2.imwrite(path, image)

    def _cv_detect_objects(self, image) -> List[Dict]:
        """Detect objects in image"""
        results = self.object_detector(image)
        detections = []
        
        for result in results:
            for box in result.boxes:
                detections.append({
                    'label': result.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': {
                        'x': int(box.xyxy[0][0]),
                        'y': int(box.xyxy[0][1]),
                        'width': int(box.xyxy[0][2] - box.xyxy[0][0]),
                        'height': int(box.xyxy[0][3] - box.xyxy[0][1])
                    }
                })
        
        return detections

    def _cv_classify_image(self, image) -> str:
        """Classify image"""
        result = self.classifier(image)
        return result[0]['label']

    def _cv_detect_faces(self, image) -> List[Dict]:
        """Detect faces in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        
        return [{
            'bbox': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            }
        } for (x, y, w, h) in faces]

    def _cv_extract_text(self, image) -> str:
        """Extract text from image"""
        return pytesseract.image_to_string(image)

    def _cv_generate_image(self, prompt: str) -> Any:
        """Generate image from prompt"""
        image = self.image_generator(prompt).images[0]
        return np.array(image)

    def _cv_draw_rectangle(self, image, x: int, y: int, width: int, height: int):
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)


    def _execute_Literal(self, node: Literal) -> Any:
        """Execute literal value"""
        return node.value

    def _execute_Variable(self, node: Variable) -> Any:
        """Execute variable reference"""
        return self.current_env.get(node.name)

    def _execute_Return(self, node: Return) -> Any:
        """Execute return statement"""
        if node.value:
            return self.execute(node.value)
        return None

    def _execute_function_call(self, func: FunctionDeclaration, args: List[Any]) -> Any:
        """Helper method to execute function call"""
        if len(args) != len(func.parameters):
            raise VexniRuntimeError(
                f"Expected {len(func.parameters)} arguments but got {len(args)}"
            )

        # Create new environment for function
        func_env = Environment({}, {}, self.global_env)
        
        # Bind parameters to arguments
        for (param_name, _), arg in zip(func.parameters, args):
            func_env.define(param_name, arg)

        # Execute function body
        prev_env = self.current_env
        self.current_env = func_env
        
        return_value = None
        try:
            for stmt in func.body:
                if isinstance(stmt, Return):
                    return_value = self.execute(stmt.value) if stmt.value else None
                    break
                self.execute(stmt)
        finally:
            self.current_env = prev_env
            
        return return_value
    
    def _execute_IndexOperation(self, node: IndexOperation) -> Any:
        target_value = self.execute(node.target)
        index_value = self.execute(node.index)

        # If target_value is a Python dict or list, simply index into it:
        if isinstance(target_value, dict):
            if index_value in target_value:
                return target_value[index_value]
            else:
                raise VexniRuntimeError(f"Key '{index_value}' not found in dictionary")
        elif isinstance(target_value, list):
            if isinstance(index_value, int) and 0 <= index_value < len(target_value):
                return target_value[index_value]
            else:
                raise VexniRuntimeError(f"Index '{index_value}' out of range for list")
        else:
            raise VexniRuntimeError(f"Cannot index into type {type(target_value)}")
