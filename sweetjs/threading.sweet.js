macro (->) {
    rule {
        $x:expr,
        $fun1(),
        $($fun2($z (,) ...)) (,) ...
    } => {
        (-> $fun1($x),
            $($fun2($z (,) ...)) (,) ...)
    }
    rule {
        $x:expr,
        $fun1($y (,) ...),
        $($fun2($z (,) ...)) (,) ...
    } => {
        (-> $fun1($x, $y (,) ...),
            $($fun2($z (,) ...)) (,) ...)
    }
    rule {
        $x:expr,
        $fun()
    } => {
        $fun($x)
    }
    rule {
        $x:expr,
        $fun($y (,) ...)
    } => {
        $fun($x, $y (,) ...)
    }
    rule {
        $x:expr
    } => {
        $x
    }
}

macro (->>) {
    rule {
        $x:expr,
        $fun1(),
        $($fun2($z (,) ...)) (,) ...
    } => {
        (->> $fun1($x),
            $($fun2($z (,) ...)) (,) ...)
    }
    rule {
        $x:expr,
        $fun1($y (,) ...),
        $($fun2($z (,) ...)) (,) ...
    } => {
        (->> $fun1($y (,) ..., $x),
            $($fun2($z (,) ...)) (,) ...)
    }
    rule {
        $x:expr,
        $fun()
    } => {
        $fun($x)
    }
    rule {
        $x:expr,
        $fun($y (,) ...)
    } => {
        $fun($y (,) ..., $x)
    }
    rule {
        $x:expr
    } => {
        $x
    }
}

var map = function(f, coll) {
    return coll.map(f);
};

var filter = function(f, coll) {
    return coll.filter(f);
};


var plus1 = function(x) {
    return x+1;
};

var even = function(x) {
    return x%2 == 0;
};

(->> [2,3],
    map(plus1), 
    filter(even));
    

    
    
    