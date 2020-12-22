const isInt = (x) => typeof(x) === "string" && x.match(/-?[0-9]+/)

const isVariable = (x) => x && x.name;

const isFunction = (x) => x && x.arg && x.body;
const isFunctionType = (x) => x && Array.isArray(x)
const isFunctionApplication = (x) => x && x.function && x.arg


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
  } else if (isVariable(val)) {
    return context[val.name]
  } else if (isFunctionApplication(val)) {
    const fType = synthesize({ val: val.function, context: context});
    if (!isFunctionType(fType)) {
      throw Error(`Expected Function Type, got ${fType}`)
    }
    const [t1, t2] = fType;
    check({ 
      val: val.arg,
      type: t1,
      context: context,
    })
    return t2
  } else if (isFunction(val)) {
    if (val.arg.type) {
      const bodyType = synthesize({ 
        val: val.body,
        context: {
          ...context,
          [val.arg.name]: val.arg.type
        }
      })
      return [val.arg.type, bodyType]
    } else {
      throw Error("Can't determine type. Please add type annotation")
    }
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

runTest(() => check({
  val: "true",
  type: "boolean",
  context: {},
}))


runFail(() => check({
  val: "true",
  type: "int",
  context: {},
}))

runTest(() => synthesize({
  val: "true",
  context: {},
}))

runTest(() => synthesize({
  val: "123",
  context: {},
}))

runTest(() => check({
  val: "123",
  type: "int",
  context: {},
}))

runFail(() => check({
  val: "123",
  type: "boolean",
  context: {},
}))

runTest(() => check({
  val: {name: "x"},
  type: "boolean",
  context: {"x": "boolean"}
}))

runFail(() => check({
  val: {name: "x"},
  type: "boolean",
  context: {"x": "int"}
}))


runTest(() => synthesize({
  val: {name: "x"},
  context: {"x": "boolean"}
}))

runTest(() => synthesize({
  val: {name: "x"},
  context: {"x": "int"}
}))


runTest(() => check({
  val: {
    arg: {name: "x"},
    body: "true"
  },
  type: ["int", "boolean"]
}))

runTest(() => check({
  val: {
    arg: {name: "x"},
    body: {name: "x"}
  },
  type: ["int", "int"]
}))

runFail(() => check({
  val: {
    arg: {name: "x"},
    body: {name: "x"}
  },
  type: ["int", "boolean"]
}))

runTest(() => synthesize({
  val: {
    function: {name: "f"},
    arg: "123"
  },
  context: {
    "f": ["int", "int"]
  }
}))

runFail(() => synthesize({
  val: {
    function: {name: "f"},
    arg: "true"
  },
  context: {
    "f": ["int", "int"]
  }
}))


runTest(() => check({
  val: {
    function: {
      arg: {name: "x", type: "int"},
      body: {name: "x"}
    },
    arg: "123"
  },
  context: {},
  type: "int"
}))




