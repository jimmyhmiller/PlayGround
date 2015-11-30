let typeof = macro {
    rule {
        NaN
    } => {
        "NaN"
    }
    rule {
        $x:expr
    } => {
        typeof $x
    }
}

typeof 2
typeof NaN === "number"

/////////////////////////////////////////////


let function = macro {

    rule {
        ($x (,) ...) {
            $body...
        }
    } => {
        function ($x (,) ...) {
            var definedArgs = [$x (,) ...]
            var args = Array.slice(arguments);
            if (args.length != definedArgs.length) {
                throw Error("function requires " + definedArgs.length + " argument(s), you supplied " + args.length)
            }
            $body...
        }
    }
}


let x = function(y) {
    return y;
}

x(2,3)
x(2)






