var mori = {};
macro to_str {
  case { _ ($toks ...) } => {
    return [makeValue(#{ $toks ... }.map(unwrapSyntax).join(''), #{ here })];
  }
}

macro (#) {
    rule {
        [$x:expr (,) ...]
    } => {
        mori.vector($x (,) ...)
    }
    rule {
        [$x:expr ...]
    } => {
        mori.vector($x (,) ...)
    }
    rule {
        ($x:expr (,) ...)
    } => {
        mori.list($x (,) ...)
    }
    rule {
        ($x:expr ...)
    } => {
        mori.list($x (,) ...)
    }
    rule {
        {$x:expr (,) ...}
    } => {
        mori.map($x (,) ...)
    }
    rule {
        {$x:expr ...}
    } => {
        mori.map($x (,) ...)
    }
}




macro (:) {
    rule {
        $x
    } => {
        to_str($x)
    }
}

macro (==) {
    rule infix {
        $x:expr | $y:expr
    } => {
        $x === $y
    }
}

macro var {
    rule {
        $name:ident = $x:expr
    } => {
        let $name = $x
    }
}

macro (::) {
    rule infix {
        $obj:expr | $fn($args:expr (,) ...)
    } => {
        $fn($obj, $args (,) ...)
    }
    rule infix {
        $obj:expr | $fn
    } => {
        $fn($obj)
    }
}

macro (:::) {
    rule infix {
        $obj:expr | $fn($args:expr (,) ...)
    } => {
        $fn($args (,) ..., $obj)
    }
    rule infix {
        $obj:expr | $fn
    } => {
        $fn($obj)
    }
}

macro (|>) {
    rule infix {
        $obj:expr | $stuff
    } => {
        $obj :: $stuff
    }
}

macro (|>>) {
    rule infix {
        $obj:expr | $stuff
    } => {
        $obj ::: $stuff
    }
}


let (+) = macro {
    rule infix {
        $x:expr | $y:expr
    } => {
        $x + $y
    }
    rule {
        $x
    } => {
        (y) => $x + y
    }
}

macro (@) {
    rule {
        $fnName:ident
        function $name($args (,) ...) {$body ...}
    } => {
        function $name($args (,) ...) {$body ...}
        $name = $fnName($name);
    }
}

var stuff = (f) => (x) => {
    console.log(x);
    return f(x);
}

@stuff 
function test(x,y) {
 return 3+3
}

let (-) = macro {
    rule infix {
        $x:expr | $y:expr
    } => {
        $x - $y
    }
     rule {
        $x
    } => {
        (y) => $x - y
    }
}

(+ 2)(3)


if (2 == '2') {

}



var x = #{
    :stuff :jimmy
    :hello :hi
}

#[1 2 3]
    :::map(+2)
    :::filter(even)
    :::map(+3)
    
    
 #[1 2 3]
    |>> map(+2)
    |>> filter(even)
    |>> map(+3)
    
    
    
    
    
    
    
