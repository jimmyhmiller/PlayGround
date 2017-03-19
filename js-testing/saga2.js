import { parse, print as toCodePrime } from 'recast';
import { transform } from 'regenerator';
import multi from 'multiple-methods';
import _ from 'lodash';
import traverse from 'babel-traverse';


const toCode = node => toCodePrime(node).code;

const log = (...args) => {
    console.log(...args);
    return args[args.length - 1];
}

const print = (arg) => {
    return log(JSON.stringify(arg, null, 2))
}

const es6Source = `
function* fetchUser() {
    yield take('stuff')
    if (x == 2) {
       yield take('stuff') 
    } else if (test) {
        yield take('stuff') 
    } else {
        yield take('stuff') 
    }
    try {
        yield put('test');
        yield call('test');
    } catch (e) {
        yield put('error');
    }
    yield take('otherStuff')
}
`

const ast = parse(es6Source);
const generatorAst = transform(ast);

const contextNext = path => {
  console.log('node', path.node);
  if (path.node.left.object.name === '_context' && path.node.left.property === 'next') {
    return path.node.right.value;
  }
}

const extractSwitch = path => {
  console.log('switch')
  const nexts = reduce(evaluate, path.node.consequent, path)
  console.log('nexts');
  return path.node.test.value;
}

const evaluate = multi(ast => ast.node.type);
evaluate.method('SwitchCase', extractSwitch);
evaluate.method('AssignmentExpression', contextNext);

evaluate.defaultMethod((ast) => {
  console.log(ast)
})

function reduce(f, ast, path) {
    const nodes = []
    traverse(ast, {
        enter: path => {
            console.log('path', path)
            const result = f(path);
            if (result) {
                nodes.push(result);
                return path.skip();
            }
        }
    }, undefined, undefined, path)
    return nodes;
}



print(reduce(evaluate, generatorAst)
    .filter(_.identity))

console.log('\n\n\n\n\n\n\n\n\n\n\n\n')

