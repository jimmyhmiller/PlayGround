

macroclass bind {
  pattern {
    rule { 
        $x:ident <- $y:expr;
    }
  }
  pattern {
    rule {
        $x:ident <- $y:expr
    }
  }
}

macroclass bindAll {
  pattern {
    rule { 
        [$x:ident (,) ...] <- [$y:expr (,) ...];
    }
  }
  pattern {
    rule {
        [$x:ident (,) ...] <- [$y:expr (,) ...]
    }
  }
}





macro do {

    rule {
        {
            $b:bindAll
            $z:expr;
        }
    } => {
        $q.all([$b$y (,) ...]).then(_.spread(function ($b$x ...) {
            return $z;
        }))
    }
    rule {
        {
            $b:bindAll
            $c ...
            $z:expr;
        }
    } => {
        $q.all([$b$y (,) ...]).then(_.spread(function ($b$x ...) {
            return do { 
                $c ...
                $z;
            }
        }))
    }

    rule {
        {
            $a:bind
            $b ...
            $c:expr;
        }
    } => {
        $a$y.then(function($a$x) {
            return do { 
                $b ...
                $c;
            }
        })
    }
    rule {
        {
            $b:bind
            $z:expr;
        }
    } => {
        $b$y.then(function($b$x) {
            return $z;
        })
    }

    rule {
        {
            $z:expr;
        }
    } => {
        $z
    }
}
let pure = (x) => new Promise((resolve, reject) => resolve(x));

do {
    x <- pure(2);
    y <- pure(2);
    z <- pure(2);
    a <- pure(3);
    b <- pure(3);
    x + y + z + a + b;
};

do {
    a <- pure(3);
    [x, y] <- [pure(2), pure(3)];
    z <- pure(3);
    x + y + z;
}
















