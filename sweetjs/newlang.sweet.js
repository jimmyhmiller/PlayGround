
macro to_str {
  case { _ ($toks ...) } => {
    return [makeValue(#{ $toks ... }.map(unwrapSyntax).join(''), #{ here })];
  }
}

macro class {
    rule {
        $name() 
            $body
    } => {
        var $name = function() { 
            var that = this;
            var handler = {
                get: (target, name) => {
                    if (name in target) {
                        return target[name]
                    }
                    return () => $name._methods[name](new Proxy(target, this))
                }
            }
            this.self = $body
            return new Proxy(this.self, handler)
        }
        $name._methods = {}
        
    }
    rule {
        $name($x (,) ...) | $guard:expr = $y
            
            $body
        
    } => {
        var $name = function($x (,) ...) {
          if($guard) {
              return $y
          }
        this.self = $body
        return this.self
        }
    }
    
    rule {
        $name($x (,) ...)
            $body
        
    } => {
        var $name = function($x (,) ...) {
            var that = this;
            var handler = {
                get: (target, name) => {
                    if (name in target) {
                        return target[name]
                    }
                    return () => $name._methods[name](new Proxy(target, this))
                }
            }
            this.self = $body
            return new Proxy(this.self, handler)
        }
        $name._methods = {}
    }

}


macro True {
    rule { }
    => {
        true
    }
}

macro False {
    rule { }
    => {
        false
    }
}


let self = macro {
    rule {
    } => {
        this.self
    }
}


macro (::) {
    rule infix {
        $x($a (,) ...) | $y:expr
    } => {
        ($y[to_str($x)]($a (,) ...))
    }
}

macro module {
    rule {
        $name {
            $body ...
        }
    } => {
        var $name = (function () {
            $body ...
        })()
    }
}

macro extend {
    rule {
        $name {
            $($type $[:] {
                $methodName $[:] $fn:expr
            }) (,) ...
        }
    } => {
        $($name.$type._methods.$methodName = $fn)...
    }
}


module List {
    class Empty() {
        empty: () => True,
        first: () => "Error",
        rest: () => self
    };

    class Cons(x, coll) {
        empty: () => False,
        first: () => x,
        rest: () => coll
    };
    return { 
        Cons: Cons,
        Empty: Empty
    }
}


module Tree {
    class Empty() {
        empty: () => True,
        left: () => "Empty",
        right: () => "Empty"
    }

    class Node(l, n, r) {
        empty: () => False,
        left: () => l,
        right: () => r
    }
    return {
        Empty: Empty,
        Node: Node
    }
}


extend List {
    Empty: {
        last: (s) => "Error"
    },
    Cons: {
        last: (s) => {
            if (s.rest().empty()) {
                return s.first();
            }
            return s.rest().last();
        }
    }
}


var list = () => {
    var result = List.Empty()
    Array.prototype.slice.call(arguments).reverse().forEach(a => {
        result = List.Cons(a, result);
    });
    return result;
}

console.log(list(1,2,3).last())


