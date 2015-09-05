

macroclass method {
  pattern {
    rule { 
        $name($args (,) ...);
    }
  }
    pattern {
    rule { 
        $name($args (,) ...)
    }
  }
}


macro protocol {
    rule {
        $name {
            $method:method...
        }
    } => {
        const $name = pprotocol({
          $($method$name: function($method$args (,) ...){
            // returns true if string matches pattern
          })(,) ...
        })

    }
}


macro implements {
    rule infix {
        $class | $proto {
            $($method:method {
                $body...
            })...
        }
    } => {
        $proto.implementation($class, {
            $($method$name: function($method$args (,) ...) {
                $body...
            }) (,) ...
        })

    }
}


User implements Equal {
  equals(one, other) {
    return one.id === other.id;
  }
};




protocol Equal {
  equals(one, other);
  equals2(one, other);
}

