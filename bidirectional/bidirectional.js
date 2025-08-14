const isInt = (x) => typeof(x) === "string" && x.match(/-?[0-9]+/)
const isString = (x) => typeof(x) === "string" && x.startsWith('"') && x.endsWith('"')

const isVariable = (x) => x && x.name;

const isFunction = (x) => x && x.arg && x.body;
const isFunctionType = (x) => x && Array.isArray(x)
const isFunctionApplication = (x) => x && x.function && x.arg
const isArithmeticOp = (x) => x && x.op && (x.op === "+" || x.op === "-" || x.op === "*" || x.op === "/") && x.left && x.right
const isIfExpression = (x) => x && x.condition && x.then && x.else !== undefined
const isConstructor = (x) => x && x.constructor && x.args
const isMatch = (x) => x && x.match && x.expr && Array.isArray(x.cases)
const isTypeVariable = (x) => typeof(x) === "string" && x.startsWith("'") && x.length > 1

// Global type environment for user-defined types
const typeEnv = {};

const collectTypeVars = (type) => {
  if (isTypeVariable(type)) {
    return [type];
  } else if (Array.isArray(type)) {
    return [...new Set(type.flatMap(collectTypeVars))];
  }
  return [];
}

const isGenericFunctionType = (x) => x && Array.isArray(x) && collectTypeVars(x).length > 0


const substitute = (type, substitutions) => {
  if (isTypeVariable(type)) {
    return substitutions[type] || type;
  } else if (Array.isArray(type)) {
    return type.map(t => substitute(t, substitutions));
  }
  return type;
}

const unify = (type1, type2, substitutions = {}) => {
  type1 = substitute(type1, substitutions);
  type2 = substitute(type2, substitutions);
  
  if (type1 === type2) {
    return substitutions;
  } else if (isTypeVariable(type1)) {
    return { ...substitutions, [type1]: type2 };
  } else if (isTypeVariable(type2)) {
    return { ...substitutions, [type2]: type1 };
  } else if (Array.isArray(type1) && Array.isArray(type2) && type1.length === type2.length) {
    let newSubs = substitutions;
    for (let i = 0; i < type1.length; i++) {
      newSubs = unify(type1[i], type2[i], newSubs);
    }
    return newSubs;
  } else {
    throw Error(`Cannot unify ${JSON.stringify(type1)} with ${JSON.stringify(type2)}`);
  }
}

const typeError = ({ val, type, context }) => {
  if (isVariable(val)) {
    throw Error(`Got variable ${val.name} of type ${context[val.name]}, expected ${type}`)
  }
  throw Error(`Got ${JSON.stringify(val)}, expected ${type}`)
}

const synthesize = ({ val, context }) => {
  if (val === "true" || val === "false") {
    return "boolean";
  } else if (isInt(val)) {
    return "int";
  } else if (isString(val)) {
    return "string";
  } else if (isVariable(val)) {
    return context[val.name]
  } else if (isFunctionApplication(val)) {
    const fType = synthesize({ val: val.function, context: context});
    if (isGenericFunctionType(fType)) {
      // For generic function application, we need to infer type arguments
      const [t1, t2] = fType;
      const argType = synthesize({ val: val.arg, context: context });
      
      // Unify the argument type with the parameter type to infer type variables
      const substitutions = unify(t1, argType);
      const resultType = substitute(t2, substitutions);
      return resultType;
    } else if (isFunctionType(fType)) {
      const [t1, t2] = fType;
      check({ 
        val: val.arg,
        type: t1,
        context: context,
      })
      return t2;
    } else {
      throw Error(`Expected Function Type, got ${fType}`)
    }
  } else if (isFunction(val)) {
    if (val.arg.type) {
      const bodyType = synthesize({ 
        val: val.body,
        context: {
          ...context,
          [val.arg.name]: val.arg.type
        }
      })
      const functionType = [val.arg.type, bodyType];
      return functionType;
    } else {
      throw Error("Can't determine type. Please add type annotation")
    }
  } else if (isArithmeticOp(val)) {
    const leftType = synthesize({ val: val.left, context: context });
    const rightType = synthesize({ val: val.right, context: context });
    
    if (val.op === "+") {
      // + can work with both int+int -> int and string+string -> string
      if (leftType === "int" && rightType === "int") {
        return "int";
      } else if (leftType === "string" && rightType === "string") {
        return "string";
      } else {
        throw Error(`+ requires matching operand types (int+int or string+string), got ${leftType} and ${rightType}`)
      }
    } else {
      // -, *, / only work with ints
      if (leftType !== "int" || rightType !== "int") {
        throw Error(`${val.op} requires int operands, got ${leftType} and ${rightType}`)
      }
      return "int"
    }
  } else if (isIfExpression(val)) {
    const conditionType = synthesize({ val: val.condition, context: context });
    if (conditionType !== "boolean") {
      throw Error(`If condition must be boolean, got ${conditionType}`)
    }
    const thenType = synthesize({ val: val.then, context: context });
    const elseType = synthesize({ val: val.else, context: context });
    if (thenType !== elseType) {
      throw Error(`If branches must have same type, got ${thenType} and ${elseType}`)
    }
    return thenType;
  } else if (isConstructor(val)) {
    // Find the constructor in the type environment
    let constructorType = null;
    let parentType = null;
    let typeInfo = null;
    
    for (const [typeName, info] of Object.entries(typeEnv)) {
      if (info.constructors[val.constructor]) {
        constructorType = info.constructors[val.constructor];
        parentType = typeName;
        typeInfo = info;
        break;
      }
    }
    
    if (!constructorType) {
      throw Error(`Unknown constructor: ${val.constructor}`);
    }
    
    // Check argument types match constructor signature
    if (val.args.length !== constructorType.length) {
      throw Error(`Constructor ${val.constructor} expects ${constructorType.length} args, got ${val.args.length}`);
    }
    
    // For generic types, infer type parameters from arguments
    let typeSubstitutions = {};
    
    for (let i = 0; i < val.args.length; i++) {
      const argType = synthesize({ val: val.args[i], context: context });
      const expectedType = constructorType[i];
      
      if (isTypeVariable(expectedType)) {
        // Type variable - infer from argument
        typeSubstitutions[expectedType] = argType;
      } else if (argType !== expectedType) {
        throw Error(`Constructor ${val.constructor} arg ${i}: expected ${expectedType}, got ${argType}`);
      }
    }
    
    // If this is a generic type, create an instantiated type name
    if (typeInfo.params.length > 0) {
      const instantiatedTypes = typeInfo.params.map(param => typeSubstitutions[param] || param);
      return `${parentType}<${instantiatedTypes.join(',')}>`;
    }
    
    return parentType;
  } else if (isMatch(val)) {
    const exprType = synthesize({ val: val.expr, context: context });
    
    // Extract base type and type parameters from instantiated type
    let baseType = exprType;
    let typeParams = [];
    
    if (exprType.includes('<')) {
      const match = exprType.match(/^([^<]+)<(.+)>$/);
      if (match) {
        baseType = match[1];
        typeParams = match[2].split(',').map(s => s.trim());
      }
    }
    
    // All case bodies must have the same type
    let resultType = null;
    
    for (const case_ of val.cases) {
      // For now, simple pattern matching - extend context with pattern variables
      let caseContext = { ...context };
      const [constructorName, ...patternVars] = case_.pattern;
      
      // Add pattern variables to context with their types
      if (typeEnv[baseType] && typeEnv[baseType].constructors[constructorName]) {
        const constructorTypes = typeEnv[baseType].constructors[constructorName];
        
        for (let i = 0; i < patternVars.length; i++) {
          let variableType = constructorTypes[i];
          
          // Substitute type parameters with concrete types
          if (isTypeVariable(variableType) && typeEnv[baseType].params) {
            const paramIndex = typeEnv[baseType].params.indexOf(variableType);
            if (paramIndex >= 0 && typeParams[paramIndex]) {
              variableType = typeParams[paramIndex];
            }
          }
          
          caseContext[patternVars[i]] = variableType;
        }
      }
      
      const caseType = synthesize({ val: case_.body, context: caseContext });
      
      if (resultType === null) {
        resultType = caseType;
      } else if (resultType !== caseType) {
        throw Error(`Match cases must have same type, got ${resultType} and ${caseType}`);
      }
    }
    
    return resultType;
  } else {
    throw Error("Unimplemented")
  }
}

const matchesType = (expectedType, {val, type, context }) => {
  if (expectedType === type) {
    return context
  }
  typeError({ val, type, context })
}


const check = (expr) => {
  const { val, type, context } = expr;
  if (!type) {
    throw Error("No type given to check", expr)
  }
  if (val === "true" || val === "false") {
    return matchesType("boolean", expr)
  } else if (isInt(val)) {
    return matchesType("int", expr)
  } else if (isString(val)) {
    return matchesType("string", expr)
  } else if (isVariable(val)) {
    return matchesType(context[val.name], expr)
  } else if (isFunction(val)) {
    if (!isFunctionType(type)) {
      throw Error(`Expected Function Type, got ${type}`)
    }
    const newContext = check({
      val: val.body,
      type: type[1],
      context: {
        ...context,
        [val.arg.name]: type[0]
      }
    })
    return newContext
  } else if (isArithmeticOp(val)) {
    if (val.op === "+") {
      // + can return int or string
      if (type === "int") {
        check({ val: val.left, type: "int", context: context });
        check({ val: val.right, type: "int", context: context });
      } else if (type === "string") {
        check({ val: val.left, type: "string", context: context });
        check({ val: val.right, type: "string", context: context });
      } else {
        throw Error(`+ returns int or string, expected ${type}`)
      }
    } else {
      // -, *, / only return int
      if (type !== "int") {
        throw Error(`${val.op} returns int, expected ${type}`)
      }
      check({ val: val.left, type: "int", context: context });
      check({ val: val.right, type: "int", context: context });
    }
    return context
  } else if (isIfExpression(val)) {
    check({ val: val.condition, type: "boolean", context: context });
    check({ val: val.then, type: type, context: context });
    check({ val: val.else, type: type, context: context });
    return context;
  } else if (isConstructor(val)) {
    // Check constructor creates value of expected type
    const constructorType = synthesize({ val: val, context: context });
    if (constructorType !== type) {
      throw Error(`Constructor creates ${constructorType}, expected ${type}`);
    }
    return context;
  } else if (isMatch(val)) {
    // Check match expression produces expected type
    const matchType = synthesize({ val: val, context: context });
    if (matchType !== type) {
      throw Error(`Match produces ${matchType}, expected ${type}`);
    }
    return context;
  } else {
    const synthType = synthesize({ val, context });
    if (synthType === type) {
      return context
    } else {
      throw Error(`Expected ${type}, got ${synthType}`)
    }
  }
}





const runTest = (testCase, shouldFail=false) => {
  try {
    const x = testCase()
    console.log(x)
    if (shouldFail) {
      console.error("Should fail, but instead got", x)
    }
    return x
  } catch (e) {
    if (shouldFail) {
      return true
    }
    console.error(e, "Shouldn't have failed")
  }
}

const runFail = (testCase) => runTest(testCase, true);

const assertEquals = (actual, expected, testName = "") => {
  const actualStr = JSON.stringify(actual);
  const expectedStr = JSON.stringify(expected);
  if (actualStr === expectedStr) {
    console.log(`✓ ${testName}: ${expectedStr}`);
    return actual;
  } else {
    console.error(`✗ ${testName}: expected ${expectedStr}, got ${actualStr}`);
    throw Error(`Assertion failed: expected ${expectedStr}, got ${actualStr}`);
  }
}

assertEquals(check({
  val: "true",
  type: "boolean",
  context: {},
}), {}, "check boolean literal")


runFail(() => check({
  val: "true",
  type: "int",
  context: {},
}))

assertEquals(synthesize({
  val: "true",
  context: {},
}), "boolean", "synthesize boolean literal")

assertEquals(synthesize({
  val: "123",
  context: {},
}), "int", "synthesize int literal")

assertEquals(check({
  val: "123",
  type: "int",
  context: {},
}), {}, "check int literal")

runFail(() => check({
  val: "123",
  type: "boolean",
  context: {},
}))

assertEquals(check({
  val: {name: "x"},
  type: "boolean",
  context: {"x": "boolean"}
}), {"x": "boolean"}, "check variable")

runFail(() => check({
  val: {name: "x"},
  type: "boolean",
  context: {"x": "int"}
}))


assertEquals(synthesize({
  val: {name: "x"},
  context: {"x": "boolean"}
}), "boolean", "synthesize boolean variable")

assertEquals(synthesize({
  val: {name: "x"},
  context: {"x": "int"}
}), "int", "synthesize int variable")


assertEquals(check({
  val: {
    arg: {name: "x"},
    body: "true"
  },
  type: ["int", "boolean"]
}), {"x": "int"}, "check function int->boolean")

assertEquals(check({
  val: {
    arg: {name: "x"},
    body: {name: "x"}
  },
  type: ["int", "int"]
}), {"x": "int"}, "check identity function")

runFail(() => check({
  val: {
    arg: {name: "x"},
    body: {name: "x"}
  },
  type: ["int", "boolean"]
}))

assertEquals(synthesize({
  val: {
    function: {name: "f"},
    arg: "123"
  },
  context: {
    "f": ["int", "int"]
  }
}), "int", "function application")

runFail(() => synthesize({
  val: {
    function: {name: "f"},
    arg: "true"
  },
  context: {
    "f": ["int", "int"]
  }
}))


assertEquals(check({
  val: {
    function: {
      arg: {name: "x", type: "int"},
      body: {name: "x"}
    },
    arg: "123"
  },
  context: {},
  type: "int"
}), {}, "check lambda application")

// Arithmetic operation tests
assertEquals(synthesize({
  val: {
    op: "+",
    left: "123",
    right: "456"
  },
  context: {}
}), "int", "synthesize int addition")

assertEquals(check({
  val: {
    op: "+",
    left: "123",
    right: "456"
  },
  type: "int",
  context: {}
}), {}, "check int addition")

runFail(() => check({
  val: {
    op: "+",
    left: "123",
    right: "456"
  },
  type: "boolean",
  context: {}
}))

assertEquals(synthesize({
  val: {
    op: "*",
    left: {name: "x"},
    right: "2"
  },
  context: {"x": "int"}
}), "int", "synthesize multiplication with variable")

runFail(() => synthesize({
  val: {
    op: "+",
    left: "123",
    right: "true"
  },
  context: {}
}))

assertEquals(check({
  val: {
    op: "-",
    left: {
      op: "+",
      left: "10",
      right: "5"
    },
    right: "3"
  },
  type: "int",
  context: {}
}), {}, "check nested arithmetic")

// String operation tests
assertEquals(synthesize({
  val: '"hello"',
  context: {}
}), "string", "synthesize string literal")

assertEquals(check({
  val: '"world"',
  type: "string",
  context: {}
}), {}, "check string literal")

assertEquals(synthesize({
  val: {
    op: "+",
    left: '"hello"',
    right: '"world"'
  },
  context: {}
}), "string", "synthesize string concatenation")

assertEquals(check({
  val: {
    op: "+",
    left: '"hello"',
    right: '"world"'
  },
  type: "string",
  context: {}
}), {}, "check string concatenation")

runFail(() => synthesize({
  val: {
    op: "+",
    left: '"hello"',
    right: "123"
  },
  context: {}
}))

runFail(() => check({
  val: {
    op: "+",
    left: '"hello"',
    right: '"world"'
  },
  type: "int",
  context: {}
}))

runFail(() => synthesize({
  val: {
    op: "*",
    left: '"hello"',
    right: '"world"'
  },
  context: {}
}))

// Generic function tests
assertEquals(synthesize({
  val: {
    arg: {name: "x", type: "'a"},
    body: {name: "x"}
  },
  context: {}
}), ["'a", "'a"], "synthesize generic identity function")

assertEquals(check({
  val: {
    arg: {name: "x", type: "'a"},
    body: {name: "x"}
  },
  type: ["'a", "'a"],
  context: {}
}), {"x": "'a"}, "check generic identity function")

assertEquals(synthesize({
  val: {
    function: {name: "identity"},
    arg: "123"
  },
  context: {
    "identity": ["'a", "'a"]
  }
}), "int", "generic identity with int")

assertEquals(synthesize({
  val: {
    function: {name: "identity"},
    arg: '"hello"'
  },
  context: {
    "identity": ["'a", "'a"]
  }
}), "string", "generic identity with string")

assertEquals(synthesize({
  val: {
    arg: {name: "x", type: "'a"},
    body: {
      arg: {name: "y", type: "'b"},
      body: {name: "x"}
    }
  },
  context: {}
}), ["'a", ["'b", "'a"]], "synthesize higher-order generic function")

// Test what happens with x + x where x is generic
runFail(() => synthesize({
  val: {
    arg: {name: "x", type: "'a"},
    body: {
      op: "+",
      left: {name: "x"},
      right: {name: "x"}
    }
  },
  context: {}
}))

// Minimal Lisp parser
const tokenize = (code) => {
  return code.replace(/\(/g, ' ( ').replace(/\)/g, ' ) ').replace(/\[/g, ' [ ').replace(/\]/g, ' ] ').replace(/:/g, ' : ').replace(/,/g, ' , ').trim().split(/\s+/).filter(x => x);
}

const parse = (tokens) => {
  if (tokens.length === 0) throw Error("Unexpected EOF");
  const token = tokens.shift();
  if (token === '(' || token === '[') {
    const list = [];
    const closeToken = token === '(' ? ')' : ']';
    while (tokens[0] !== closeToken) {
      list.push(parse(tokens));
    }
    tokens.shift(); // consume closing token
    return list;
  } else if (token === ')' || token === ']') {
    throw Error(`Unexpected ${token}`);
  } else {
    return token;
  }
}

const lispToAst = (expr) => {
  if (typeof expr === 'string') {
    if (expr === 'true' || expr === 'false') return expr;
    if (expr.match(/^-?\d+$/)) return expr;
    if (expr.startsWith('"') && expr.endsWith('"')) return expr;
    return {name: expr};
  }
  
  if (Array.isArray(expr)) {
    const [op, ...args] = expr;
    
    if (op === 'fn') {
      const [argList, body] = args;
      const [arg, , type] = argList; // ignore colon
      return {
        arg: {name: arg, type: type},
        body: lispToAst(body)
      };
    }
    
    if (op === 'if') {
      const [condition, thenExpr, elseExpr] = args;
      return {
        condition: lispToAst(condition),
        then: lispToAst(thenExpr),
        else: lispToAst(elseExpr)
      };
    }
    
    if (op === 'type') {
      // (type Maybe ['a] [Some 'a] [None])
      const [typeName, typeParams, ...constructors] = args;
      const typeInfo = {
        params: typeParams || [],
        constructors: {}
      };
      
      constructors.forEach(constructor => {
        const [name, ...argTypes] = constructor;
        typeInfo.constructors[name] = argTypes;
      });
      
      typeEnv[typeName] = typeInfo;
      return { typeDefinition: typeName }; // Just a marker
    }
    
    if (op === 'make') {
      // (make Some 42)
      const [constructor, ...constructorArgs] = args;
      return {
        constructor: constructor,
        args: constructorArgs.map(lispToAst)
      };
    }
    
    if (op === 'match') {
      // (match expr [Some x] x [None] 0)
      const [expr, ...casesFlat] = args;
      const cases = [];
      for (let i = 0; i < casesFlat.length; i += 2) {
        const pattern = casesFlat[i];
        const body = casesFlat[i + 1];
        cases.push({
          pattern: pattern,
          body: lispToAst(body)
        });
      }
      return {
        match: true,
        expr: lispToAst(expr),
        cases: cases
      };
    }
    
    if (['+', '-', '*', '/'].includes(op)) {
      const [left, right] = args;
      return {
        op: op,
        left: lispToAst(left),
        right: lispToAst(right)
      };
    }
    
    // Function application: (f arg)  
    return {
      function: lispToAst(op),
      arg: lispToAst(args[0])
    };
  }
  
  return expr;
}

const evalLisp = (code) => {
  const tokens = tokenize(code);
  const sexpr = parse(tokens);
  return lispToAst(sexpr);
}

// Lisp syntax tests
assertEquals(synthesize({
  val: evalLisp('(+ 10 20)'),
  context: {}
}), "int", "Lisp addition")

assertEquals(synthesize({
  val: evalLisp('(fn [x: int] x)'),
  context: {}
}), ["int", "int"], "Lisp identity function")

assertEquals(synthesize({
  val: evalLisp('((fn [x: int] x) 42)'),
  context: {}
}), "int", "Lisp function application")

assertEquals(synthesize({
  val: evalLisp(`(fn [x: 'a] x)`),
  context: {}
}), ["'a", "'a"], "Lisp generic identity")

assertEquals(synthesize({
  val: evalLisp('(+ "hello" "world")'),
  context: {}
}), "string", "Lisp string concatenation")

// Complex arithmetic expression with nested operations
const complexArithmetic = `(+ (* 5 (- 10 3)) (+ 20 2))`

assertEquals(synthesize({
  val: evalLisp(complexArithmetic),
  context: {}
}), "int", "Complex nested arithmetic")

// Demonstrate polymorphic functions with explicit applications
assertEquals(synthesize({
  val: evalLisp(`((fn [x: 'a] x) 42)`),
  context: {}
}), "int", "Generic identity applied to int")

assertEquals(synthesize({
  val: evalLisp(`((fn [x: 'a] x) "Hello")`),
  context: {}
}), "string", "Generic identity applied to string")

// Chain of function applications
assertEquals(synthesize({
  val: evalLisp(`
    ((fn [x: int] (+ x 5))
     ((fn [y: int] (* y 2)) 
      ((fn [z: int] (- z 1)) 10)))
  `),
  context: {}
}), "int", "Chained function applications: ((10-1)*2)+5")

// Curried function example
assertEquals(synthesize({
  val: evalLisp(`
    (fn [x: int] 
      (fn [y: int] 
        (fn [z: int] 
          (+ (+ x y) z))))
  `),
  context: {}
}), ["int", ["int", ["int", "int"]]], "Curried triple addition function")

// If expression tests
assertEquals(synthesize({
  val: evalLisp(`(if true 42 0)`),
  context: {}
}), "int", "Simple if expression")

assertEquals(synthesize({
  val: evalLisp(`(if false "hello" "world")`),
  context: {}
}), "string", "If with strings")

assertEquals(check({
  val: evalLisp(`(if true 42 0)`),
  type: "int",
  context: {}
}), {}, "Check if expression")

// If with variables
assertEquals(synthesize({
  val: evalLisp(`(if x 10 20)`),
  context: {"x": "boolean"}
}), "int", "If with boolean variable")

// Complex nested if
assertEquals(synthesize({
  val: evalLisp(`(if true (+ 10 5) (- 20 3))`),
  context: {}
}), "int", "If with arithmetic in branches")

// If in function
assertEquals(synthesize({
  val: evalLisp(`(fn [x: boolean] (if x 1 0))`),
  context: {}
}), ["boolean", "int"], "Function with if expression")

// Test if condition type checking fails
runFail(() => synthesize({
  val: evalLisp(`(if 42 "yes" "no")`),
  context: {}
}))

// Test if branch type mismatch fails  
runFail(() => synthesize({
  val: evalLisp(`(if true 42 "hello")`),
  context: {}
}))

// User-defined types tests

// Define a simple Bool type first  
evalLisp(`(type Bool [] [True] [False])`)

// Test constructor creation
assertEquals(synthesize({
  val: evalLisp(`(make True)`),
  context: {}
}), "Bool", "Constructor creates Bool type")

assertEquals(synthesize({
  val: evalLisp(`(make False)`),
  context: {}
}), "Bool", "False constructor creates Bool type")

// Test match on Bool
assertEquals(synthesize({
  val: evalLisp(`(match (make True) [True] 1 [False] 0)`),
  context: {}
}), "int", "Match on Bool type")

// Define a simple Option type with concrete int
evalLisp(`(type IntOption [] [SomeInt int] [NoneInt])`)

// Test constructor creation
assertEquals(synthesize({
  val: evalLisp(`(make SomeInt 42)`),
  context: {}
}), "IntOption", "Constructor creates IntOption type")

assertEquals(synthesize({
  val: evalLisp(`(make NoneInt)`),
  context: {}
}), "IntOption", "NoneInt constructor creates IntOption type")

// Test match expression  
assertEquals(synthesize({
  val: evalLisp(`(match (make SomeInt 42) [SomeInt x] x [NoneInt] 0)`),
  context: {}
}), "int", "Match expression extracts value")

assertEquals(synthesize({
  val: evalLisp(`(match (make NoneInt) [SomeInt x] x [NoneInt] 0)`),
  context: {}
}), "int", "Match expression handles None case")

// Function that works with IntOption
assertEquals(synthesize({
  val: evalLisp(`(fn [m: IntOption] (match m [SomeInt x] (+ x 1) [NoneInt] 0))`),
  context: {}
}), ["IntOption", "int"], "Function using IntOption type")

// Now test GENERIC ADTs!

// Define a generic Maybe type
evalLisp(`(type Maybe ['a] [Some 'a] [None])`)

// Test generic constructor creation
assertEquals(synthesize({
  val: evalLisp(`(make Some 42)`),
  context: {}
}), "Maybe<int>", "Generic constructor creates Maybe<int>")

assertEquals(synthesize({
  val: evalLisp(`(make Some "hello")`),
  context: {}
}), "Maybe<string>", "Generic constructor creates Maybe<string>")

assertEquals(synthesize({
  val: evalLisp(`(make None)`),
  context: {}
}), "Maybe<'a>", "None constructor creates Maybe<'a>")

// Test generic match expression
assertEquals(synthesize({
  val: evalLisp(`(match (make Some 42) [Some x] x [None] 0)`),
  context: {}
}), "int", "Generic match extracts correct type")

assertEquals(synthesize({
  val: evalLisp(`(match (make Some "test") [Some x] x [None] "default")`),
  context: {}
}), "string", "Generic match with strings")

// Test the problematic case: (λ b . if b then false else true) true
// This should fail because the lambda has no type annotation
runFail(() => synthesize({
  val: {
    function: {
      arg: {name: 'b'}, // Note: no type annotation
      body: {
        condition: {name: 'b'},
        then: 'false',
        else: 'true'
      }
    },
    arg: 'true'
  },
  context: {}
}))

