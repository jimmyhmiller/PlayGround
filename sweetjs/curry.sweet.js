


macro cfunction {
	case {
		_ ($x) { $body ...}
	} => {
	  	var l = #{$x}[0].token.value;
		var i = makeIdent(l, #{$x})
		letstx $i = [i]
		return #{function($i) {
			$body ...
			}
		}
	}
	
	case {
		_ ($x, $y) { $body ... }
	} => {
		var l = #{$y}[0].token.value;
		var i = makeIdent(l, #{$y})
		letstx $i = [i]
		return #{function($x, $y) {
			  if ($y == undefined) {
				return function ($i) {
					$body ...
			  }
			}
	  		$body ...
		  }
		}
	}

	case {
		_ ($x, $y, $z (,) ...) { $body ...}
	} => {
		var l = #{$y}[0].token.value;
		var i = makeIdent(l, #{$y})
		letstx $i = [i]
		return #{function($x, $y, $z (,) ...) {
			  undefinedChecks($y) {
				return cfunction($i, $z (,) ...) {
					$body ...
				}
			  }
	  		undefinedChecks($z (,) ...) {
				return cfunction($z (,) ...) {
					$body ...
				}
			  }
			  $body ...
			}
		}
	}
}

macro undefinedChecks {
	rule {
		($x) { $body ... }
	} => {
		if ($x == undefined) {
			$body ...
		}
	}
	rule {
		($x, $y (,) ...) { $body ... }
	} => {
		undefinedChecks($x) { $body ... }
		undefinedChecks($y (,) ...) { $body ...}
	}
}


let function = macro {
	rule {
	 $name($x (,) ...) { $body ... }
	 } => {
	 var $name = cfunction($x (,) ...) { $body ... }
	 }
	rule {
		($x (,) ...) {
		$body ... }
	} => {
		cfunction($x (,) ...) {
			$body ...
		}
	}
}











function add(x, y) {
	return x+y; 
}

var _ = {}

_.map = function(f, coll) {
	return coll.map(f)
}

var addmap2 = _.map(add(2))
add(2,3)
addmap2([1,2,3])







