


macro class {
    rule {
        $name() 
            $body
    } => {
        var $name = function() {
            this.self = $body
            return this.self
        }
        
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
            this.self = $body
            return this.self
        }
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












class Empty() {
    empty: () => True,
    contains: (i) => False,
    insert: (i) => Insert(self, i),
    union: (s) => s
};

class Insert(s, n) | s.contains(n) = s {
    empty: () => False,
    contains: (i) => i == n || s.contains(i),
    insert: (i) => Insert(self, i),
    union: (s) => Union(self, s)
};

class Union(s1, s2) {
    empty: () => s.empty(),
    contains: (i) => s1.contains(i) || s2.contains(i),
    insert: (i) => Insert(self, i),
    union: (s) => Union(self, s)
};













