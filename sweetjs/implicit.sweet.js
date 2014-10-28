let function = macro {
  	rule {
		($x (,) ...) { 
			$body ...
			return $return
		}
	} => {
		function($x (,) ...) {
			$body ...
			return $return
		}
	}
	rule {
		($x (,) ...) { 
			$body ...
			$return:expr
		}
	} => {
		function($x (,) ...) {
			$body ...
			return $return
		}
	}
}

var add = implictFunction(x) {
	x = 2
	x
}