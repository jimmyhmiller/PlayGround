(ns codegen.grammar)

(def grammar

"<program> = expr | comment | program*
comment = <'//'> #'.*'
phrase = phrase-type phrase-name phrase-body |
         phrase-type phrase-name phrase-args phrase-body |
         phrase-type phrase-args phrase-body
phrase-type = identifier
phrase-name = identifier
phrase-body = open-c expr* close-c
phrase-args = args


<expr> = fn-application | literal | infix-application | lambda | val | phrase | map | array

map = open-c pair* close-c
pair = expr expr
array = open-s expr* close-s


fn-application = identifier open-p (expr <','?>)* close-p
infix-application = expr <' '> symbol <' '> expr |
                    expr <' '> <'`'> identifier  <'`'> <' '> expr
<symbol> = #'[$-/:-?{-~!\"^_`\\[\\]]'
keyword = <':'> identifier
identifier = #'[A-Za-z_][A-Za-z\\-_!?0-9]*'
<literal> = number | identifier | keyword | string
number = #'[0-9]+'
string = <'\"'> #'[^\"]*' <'\"'>



fn = <'fn'> identifier open-c fn-body close-c | <'fn'> identifier lambda
fn-body = lambda+
lambda = fn-args  <'=>'> expr | fn-args <'=>'> open-c expr+ close-c
<arg> = identifier
args = open-p (arg <','?>)* close-p
fn-args = open-p ((arg | literal)  <','?>)* close-p | arg | literal

val = <'val'> identifier <'='> expr

<open-s> = <'['>
<close-s> = <']'>
<open-p> = <'('>
<close-p> = <')'>
<open-c> = <'{'>
<close-c> = <'}'>")

(def text
"
val name = \"Jimmy\"

data Action {
    Increment
    Decrement
}

data Maybe {
    None
    Just(a)
} 

data Person {
    Customer(id)
    Employee(id, position)
}

fn double(x y) { x * 2 }

fn get-customer-id {
    Customer(:id) => Just(id)
    Employee(_) => None
}

fn counter {
    (state, Increment) => state + 1
    (state, Decrement) => state - 1
    (state, _) => state
}

fn div {
    (numerator, 0) => throw DivideByZero
    (numerator, divisor) => numerator / divisor
}

protocol Fn {
    invoke(f, args)
}

implement Fn(Map) {
    invoke(map, arg) {
        get(arg, map)
    }
}



implement StateReducer(Action) {
    fn reduce(state, action) {
        match(action) {
            Increment => state + 1
            Decrement => state - 1
            otherwise => state
        }
    }
}


")
