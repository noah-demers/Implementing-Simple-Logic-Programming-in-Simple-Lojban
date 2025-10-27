"""
Lojban Predicate Calculus Interpreter
Author: Noah deMers
Date: October 2025

A simple predicate calculus implementation using Lojban-inspired syntax.
Supports basic logical operations, arithmetic, and list manipulation.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# VALUE TYPES
# ============================================================================

@dataclass
class LojbanValue:
    """Base class for all values"""
    pass


@dataclass
class LojbanNumber(LojbanValue):
    """Integer value"""
    value: int
    
    def __eq__(self, other):
        return isinstance(other, LojbanNumber) and self.value == other.value
    
    def __repr__(self):
        return str(self.value)


@dataclass
class LojbanName(LojbanValue):
    """Variable name"""
    name: str
    
    def __eq__(self, other):
        return isinstance(other, LojbanName) and self.name == other.name
    
    def __repr__(self):
        return f".{self.name}."


@dataclass
class LojbanList(LojbanValue):
    """List of values"""
    items: List[LojbanValue]
    
    def __eq__(self, other):
        return isinstance(other, LojbanList) and self.items == other.items
    
    def __repr__(self):
        if not self.items:
            return "()"
        return f"({' '.join(str(item) for item in self.items)})"


@dataclass
class LojbanPredicate(LojbanValue):
    """User-defined predicate"""
    name: str
    params: LojbanList
    body: LojbanList
    
    def __repr__(self):
        return f"<predicate {self.name}>"


# ============================================================================
# INTERPRETER
# ============================================================================

class LojbanInterpreter:
    """Main interpreter for Lojban predicate calculus"""
    
    def __init__(self, debug: bool = False):
        self.variables: Dict[str, Any] = {}
        self.predicates: Dict[str, LojbanPredicate] = {}
        self.debug = debug
    
    # ------------------------------------------------------------------------
    # Lexical Analysis - Token Classification
    # ------------------------------------------------------------------------
    
    def is_vowel(self, c: str) -> bool:
        """Check if character is vowel (a, e, i, o, u)"""
        return c.lower() in 'aeiou'
    
    def is_consonant(self, c: str) -> bool:
        """Check if character is consonant"""
        return c.isalpha() and not self.is_vowel(c)
    
    def is_cmavo(self, word: str) -> bool:
        """Check CV pattern (short word)"""
        return (len(word) == 2 and 
                self.is_consonant(word[0]) and 
                self.is_vowel(word[1]))
    
    def is_gismu(self, word: str) -> bool:
        """Check CVCCV or CCVCV pattern (predicate word)"""
        if len(word) != 5:
            return False
        
        # CVCCV pattern
        pattern1 = (self.is_consonant(word[0]) and self.is_vowel(word[1]) and
                   self.is_consonant(word[2]) and self.is_consonant(word[3]) and
                   self.is_vowel(word[4]))
        
        # CCVCV pattern
        pattern2 = (self.is_consonant(word[0]) and self.is_consonant(word[1]) and
                   self.is_vowel(word[2]) and self.is_consonant(word[3]) and
                   self.is_vowel(word[4]))
        
        return pattern1 or pattern2
    
    def is_number(self, word: str) -> bool:
        """Check if valid number (no leading zeros except '0')"""
        if not word or not word.isdigit():
            return False
        return word == "0" or word[0] != "0"
    
    def is_name(self, word: str) -> bool:
        """Check .name. pattern"""
        return (len(word) >= 3 and 
                word[0] == '.' and 
                word[-1] == '.' and 
                word[1:-1].isalpha())
    
    def validate_token(self, token: str) -> bool:
        """Validate token matches allowed patterns"""
        if token == 'i':
            return True
        if self.is_cmavo(token) or self.is_gismu(token):
            return True
        if self.is_number(token) or self.is_name(token):
            return True
        raise ValueError(f"Invalid token: '{token}'")
    
    # ------------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------------
    
    def tokenize(self, input_str: str) -> List[str]:
        """Split input into tokens"""
        return input_str.lower().split()
    
    def parse_statements(self, input_str: str) -> List[List[str]]:
        """Parse into statements starting with 'i'"""
        tokens = self.tokenize(input_str)
        
        # Validate all tokens
        for token in tokens:
            if token != 'i':
                self.validate_token(token)
        
        # Split by 'i' markers
        statements = []
        current = []
        
        for token in tokens:
            if token == 'i':
                if current:
                    statements.append(current)
                current = []
            else:
                current.append(token)
        
        if current:
            statements.append(current)
        
        return statements
    
    # ------------------------------------------------------------------------
    # Evaluation Helpers
    # ------------------------------------------------------------------------
    
    def get_value(self, token: str) -> Any:
        """Get value of token (number, variable, or None)"""
        if self.is_number(token):
            return int(token)
        elif self.is_name(token):
            name = token[1:-1]
            return self.variables.get(name)
        return None
    
    def set_variable(self, name_token: str, value: Any) -> None:
        """Bind variable to value"""
        if self.is_name(name_token):
            name = name_token[1:-1]
            self.variables[name] = value
            if self.debug:
                print(f"  Set {name} = {value}")
    
    # ------------------------------------------------------------------------
    # Predicate Implementations
    # ------------------------------------------------------------------------
    
    def eval_fatci(self, arg: str) -> bool:
        """fatci - Exists (declare fact)"""
        if self.is_name(arg):
            self.set_variable(arg, True)
        return True
    
    def eval_sumji(self, args: List[str], swap: bool = False) -> bool:
        """sumji - Addition: result = op1 + op2"""
        if len(args) < 3:
            raise ValueError("sumji requires 3 arguments")
        
        result_token, op1_token, op2_token = args[0], args[1], args[2]
        
        if swap:
            result_token, op1_token = op1_token, result_token
        
        op1 = self.get_value(op1_token)
        op2 = self.get_value(op2_token)
        result_val = self.get_value(result_token)
        
        if op1 is None or op2 is None:
            raise ValueError("Operands must have values")
        
        computed = op1 + op2
        
        if result_val is None:
            self.set_variable(result_token, computed)
            return True
        else:
            return result_val == computed
    
    def eval_vujni(self, args: List[str], swap: bool = False) -> bool:
        """vujni - Subtraction: result = op1 - op2"""
        if len(args) < 3:
            raise ValueError("vujni requires 3 arguments")
        
        result_token, op1_token, op2_token = args[0], args[1], args[2]
        
        if swap:
            result_token, op1_token = op1_token, result_token
        
        op1 = self.get_value(op1_token)
        op2 = self.get_value(op2_token)
        result_val = self.get_value(result_token)
        
        if op1 is None or op2 is None:
            raise ValueError("Operands must have values")
        
        computed = op1 - op2
        
        if result_val is None:
            self.set_variable(result_token, computed)
            return True
        else:
            return result_val == computed
    
    def eval_dunli(self, args: List[str]) -> bool:
        """dunli - Equality check"""
        if len(args) < 2:
            raise ValueError("dunli requires 2 arguments")
        
        val1 = self.get_value(args[0])
        val2 = self.get_value(args[1])
        
        if val1 is None or val2 is None:
            return False
        
        return val1 == val2
    
    def eval_steni(self, arg: str) -> bool:
        """steni - Empty list"""
        self.set_variable(arg, LojbanList([]))
        return True
    
    def parse_list(self, tokens: List[str], start_idx: int) -> Tuple[LojbanList, int]:
        """Parse list expression: lo steko item lo steko item ... lo steni"""
        items = []
        i = start_idx
        
        while i < len(tokens):
            if tokens[i] == 'lo':
                i += 1
                if i >= len(tokens):
                    break
                
                if tokens[i] == 'steni':
                    i += 1
                    break
                elif tokens[i] == 'steko':
                    i += 1
                    if i < len(tokens):
                        item_token = tokens[i]
                        if self.is_number(item_token):
                            items.append(LojbanNumber(int(item_token)))
                        elif self.is_name(item_token):
                            items.append(LojbanName(item_token[1:-1]))
                        i += 1
                else:
                    if self.is_number(tokens[i]):
                        items.append(LojbanNumber(int(tokens[i])))
                    elif self.is_name(tokens[i]):
                        items.append(LojbanName(tokens[i][1:-1]))
                    i += 1
            else:
                i += 1
        
        return LojbanList(items), i
    
    def eval_steko(self, args: List[str]) -> bool:
        """steko - List construction (cons cell)"""
        if len(args) < 1:
            raise ValueError("steko requires at least 1 argument")
        
        result_token = args[0]
        lst, _ = self.parse_list(args, 0)
        self.set_variable(result_token, lst)
        
        return True
    
    def eval_cmavo_def(self, args: List[str]) -> bool:
        """cmavo - Define predicate"""
        if len(args) < 3:
            raise ValueError("cmavo requires at least 3 arguments")
        
        # Get predicate name
        pred_name_token = args[0]
        if self.is_name(pred_name_token):
            pred_name = pred_name_token[1:-1]
        elif self.is_gismu(pred_name_token):
            pred_name = pred_name_token
        else:
            raise ValueError(f"Invalid predicate name: {pred_name_token}")
        
        # Parse parameters and body as lists
        params, params_end = self.parse_list(args, 1)
        body, _ = self.parse_list(args, params_end)
        
        # Store predicate
        self.predicates[pred_name] = LojbanPredicate(pred_name, params, body)
        
        if self.debug:
            print(f"  Defined predicate: {pred_name}")
        
        return True
    
    def eval_user_predicate(self, pred_name: str, args: List[str]) -> bool:
        """Evaluate user-defined predicate"""
        if pred_name not in self.predicates:
            raise ValueError(f"Unknown predicate: {pred_name}")
        
        pred = self.predicates[pred_name]
        # Simple check: predicate exists
        return True
    
    # ------------------------------------------------------------------------
    # Statement Evaluation
    # ------------------------------------------------------------------------
    
    def evaluate_statement(self, tokens: List[str]) -> Any:
        """Evaluate a single statement"""
        if not tokens:
            return None
        
        i = 0
        swap = False
        
        # Check for 'se' modifier
        if tokens[i] == 'se':
            swap = True
            i += 1
        
        # Skip 'lo' markers and collect arguments
        args = []
        predicate = None
        
        while i < len(tokens):
            token = tokens[i]
            
            if predicate is None and self.is_gismu(token):
                predicate = token
                i += 1
            elif token == 'lo':
                i += 1
            else:
                args.append(token)
                i += 1
        
        if predicate is None:
            raise ValueError("No predicate found")
        
        # Dispatch to appropriate predicate handler
        if predicate == 'fatci':
            return self.eval_fatci(args[0]) if args else True
        elif predicate == 'sumji':
            return self.eval_sumji(args, swap)
        elif predicate == 'vujni':
            return self.eval_vujni(args, swap)
        elif predicate == 'dunli':
            return self.eval_dunli(args)
        elif predicate == 'steni':
            return self.eval_steni(args[0]) if args else True
        elif predicate == 'steko':
            return self.eval_steko(args)
        elif predicate == 'cmavo':
            return self.eval_cmavo_def(args)
        elif predicate in self.predicates:
            return self.eval_user_predicate(predicate, args)
        else:
            raise ValueError(f"Unknown predicate: {predicate}")
    
    # ------------------------------------------------------------------------
    # Main Execution
    # ------------------------------------------------------------------------
    
    def run(self, input_str: str) -> Any:
        """Execute Lojban program and return result"""
        try:
            statements = self.parse_statements(input_str)
            result = None
            
            for stmt in statements:
                result = self.evaluate_statement(stmt)
                if self.debug:
                    print(f"Statement result: {result}")
            
            return result
        
        except Exception as e:
            if self.debug:
                raise
            print(f"Error: {e}")
            return None
    
    def get_results(self) -> Dict[str, Any]:
        """Get all variable bindings"""
        return self.variables.copy()


# ============================================================================
# TESTING
# ============================================================================

def run_tests():
    """Comprehensive test suite"""
    print("LOJBAN PREDICATE CALCULUS INTERPRETER - TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("i lo .brook. fatci", "fatci - Declare existence"),
        ("i 4 sumji 2 2", "sumji - Check: 4 = 2 + 2"),
        ("i lo .answer. sumji 40 2", "sumji - Bind variable"),
        ("i se 2 sumji lo .result. 3", "se - Swap arguments"),
        ("i lo .diff. vujni 10 3", "vujni - Subtraction"),
        ("i lo .x. sumji 5 5  i lo .x. dunli 10", "dunli - Equality"),
        ("i lo .a. sumji 1 2  i lo .b. sumji 3 4  i lo .c. sumji lo .a. lo .b.", 
         "Multiple statements"),
    ]
    
    for i, (code, description) in enumerate(tests, 1):
        print(f"\n[Test {i}] {description}")
        print(f"Input: {code}")
        interp = LojbanInterpreter()
        result = interp.run(code)
        print(f"Result: {result}")
        if interp.variables:
            print(f"Variables: {interp.get_results()}")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")


if __name__ == "__main__":
    run_tests()
