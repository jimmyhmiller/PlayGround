
macro arg {
    rule { 
        $arg:ident 
    } => {
        
    }
    rule { 
        ?$arg:ident 
    } => {
    
    }
    rule { 
        +$arg:ident 
    } => {
    
    }
    rule { 
        *$arg:ident 
    } => {
    
    }
}

macro method {
    rule {
        $name:ident($args (,) ...)
        $($othername:ident($otherargs (,) ...))... {
            $body...
        }
    } => {
        let $name = macro {
            rule {
                $stuff:invoke(multiNameFunction)
            } => {
                (function($args (,) ..., $($otherargs (,) ...) (,) ... ) {
                    $body...
                })$stuff
            }
        }
    }
}




macro multiNameFunction {
    rule {
        ($initArgs:expr (,) ...)$($name($args:expr (,) ...))...
    } => {
        (($initArgs (,) ...), $($args   (,) ...) (,)...) 
    }
}




method checkif(val) isbetween(low) and(high) {
    return val > low && val < high;
}

method iff(pred) then(trueCase) elses(falseCase) {
    if (pred) {
        return trueCase;
    } else {
        return falseCase;
    }
}

checkif(2) 
isbetween(1)
and(4);

iff(2==2) 
then(2)
elses(3)










