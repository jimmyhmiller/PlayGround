macro to_str {
  case { _ ($toks ...) } => {
    return [makeValue(#{ $toks ... }.map(unwrapSyntax).join(''), #{ here })];
  }
}


function ADT (constr, values) {
    return {
        constr: constr,
        values: values
    }
}

var betterEquality = function(a, b) {
    if (a instanceof ADT && b instanceof ADT) {
        return a.constr == b.constr &&
              a.values.every(function(x,i) {
                return betterEquality(x, b[i]);
              });
    }
    else {
        return a == b;
    }
}


macro fun {
    rule { 
        () ($name, $cons)
    } => {
        new ADT(to_str($name.$cons), []);
    }
    rule { 
        ($x ...) ($name, $cons)
    } => {
        function ($x (,) ...) {
            return new ADT(to_str($name.$cons), [$x (,) ...]);
        }
    }
}

macro data {
    rule {
        $name = $($cons $x:ident ...) (|) ...;
    } => {
        $(var $cons = fun ($x ...) ($name, $cons) ) ...;
        $($cons.constr = to_str($name.$cons);) ...
    }
}

macro constrValues {
    case {
        _ ($cons), $n, $x, ($body:expr)
    } => {
        return #{ return $body }
    }
    
    case {
        _ ($cons $y), $n, $x, ($body:expr)
    } => {
        var l = #{$y}[0].token.value;
        if (typeof l == "string" && l.charAt(0) != l.toUpperCase()) {
            var i = makeIdent(l, #{$y})
            letstx $i = [i]
            return #{
              return (function($i) {
                return $body
              })($x.values[$n])
            }
        }
        return #{
            if (betterEquality($x.values[$n], $y)) {
                return $body
            }
        }
    }
    
    case {
        _ ($cons $y $ys ...), $n, $x, ($body:expr)
    } => {
        var x = unwrapSyntax(#{$n});
        var y = makeValue(x+1, #{$n});
        var z = makeValue(x, #{$n});
        letstx $a = [y], $b =[z];
        return #{constrValues ($cons $y), $b, $x, 
            (function() {
                constrValues ($cons $ys ...), $a, $x, $body
                }())} 
                     
    }
    
}



macro match {
    case {
        _ ($x:expr) {
            $(($cons $y ...) => $body:expr) ...
        }
    } => {
        return #{
            (function(x) {
              $(if (x.constr == $cons.constr) {
                constrValues ($cons $y ...), 0, x, ($body)
              }) ...
              else {
                return "error"
              }
            })($x)
        }
    }
}



data Nat = Z | S n;

var isZero = function(x) {
  return match (x) {
    (Z) => true
    (S Z) => false
  }
}

var inc = function(x) {
  return match (x) {
    (Z) => S(Z)
    (S n) => S(S(n))
  }
}

var dec = function(x) {
    return match(x) {
        (Z) => "error"
        (S n) => n
    }
}

var add = function(a,b) {
    return match(b) {
        (Z) => a
        (S n) => add(inc(a), dec(b))
    }
}

var toNum = function(x) {
    return match (x) {
        (Z) => 0
        (S n) => 1 + toNum(n)
    }
}


macro thread {
    rule {
        $x $y($z (,) ...)
    } => {
        $y($x, $z (,) ...)
    }
    rule {
        $x:expr $($y($z (,) ...) ...
    } => {
        thread $(thread $x $y($z (,) ...)) ...
    }
}