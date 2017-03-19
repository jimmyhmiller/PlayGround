const parse = require('esprima').parse;
const { traverse, VisitorOption: { Skip } } = require('estraverse');
const multi = require('multiple-methods');
const _ = require('lodash/fp');
const selectn = require('selectn');
const Immutable = require('Immutable');

const log = (...args) => {
    console.log(...args);
    return args[args.length - 1];
}

const print = (arg) => {
    return log(JSON.stringify(arg, null, 2))
}



const yieldArgument = multi(_.property('type'));
yieldArgument.method('CallExpression', selectn('callee.name'));
yieldArgument.method('ArrayExpression', _.compose(_.map(yieldArgument), selectn('elements')))
yieldArgument.defaultMethod(log);

const extractYield = (yieldExpression) => {
    return yieldArgument(yieldExpression.argument);
}


const extractTry = (tryStatement) => {
    return {
        tryBranch: reduce(eval, tryStatement.block),
        catchBranch: reduce(eval, tryStatement.handler)
    }
}

const extractIf = (ifStatement) => {
    return {
        ifBranch: reduce(eval, ifStatement.consequent),
        elseBranch: reduce(eval, ifStatement.alternate)
    }
}


const eval = multi(_.property('type'));
eval.method('YieldExpression', extractYield)
eval.method('TryStatement', extractTry);
eval.method('IfStatement', extractIf)


eval.defaultMethod((ast) => undefined)

function reduce(f, ast) {
    const nodes = []
    traverse(ast, {
        enter: node => {
            const result = f(node);
            if (result) {
                nodes.push(result);
                return Skip;
            }
        }
    })
    return nodes;
}


const ast = parse(`
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
`)

// print(ast)


print(reduce(eval, ast)
    .filter(_.identity))



console.log('reload\n\n')