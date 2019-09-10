#lang slideshow

(require (planet dyoo/bf:1:8/language))


(require slideshow/repl)


(define point item)
(define bullet item)

(define (source-slide source
                      #:title [title ""]
                      #:interactive [interactive #f]
                      #:auto-eval? [auto-eval? #t]
                      #:module-name [module-name "program.rkt"]
                      #:result-height [result-height 3/10])
  (define rg (make-repl-group #:prompt (if interactive ">" #f)))
  (define backing (make-module-backing rg source  #:module-name module-name))
  (slide
   #:title title
   (module-area backing
                #:font-size 26
                #:width (* client-w 8/10)
                #:height (* client-h (- 1 result-height 1/10))
                #:auto-eval? auto-eval?)
   (result-area rg
                #:width (* client-w 8/10)
                #:height (* client-h result-height))))


(define (full-source-slide source #:title [title ""])
  (define rg (make-repl-group))
  (define backing (make-module-backing rg source))
  (slide
   #:title title
   (module-area backing
                #:font-size 26
                #:width (* client-w 9/10)
                #:height (* client-h 9/10))))


(enable-click-advance! #f)
(set-page-numbers-visible! #t)

(slide
 (titlet "A Programming-Language Programming Language")
 (bitmap (build-path (collection-path "icons") "plt.gif")))

(slide
 (titlet "These slides are a bit different, we will see why later."))

(slide
 #:title "Disclaimers"
 (item "An Introduction to Racket")
 (item "Not a tutorial")
 (item "Focus on what makes Racket unique"))

(slide
 #:title "Plan"
 (item "Racket as a regular language")
 (item "Interesting features worth exploring.")
 (item "A Programming-Language Programming Language")
 (item "Exploring languages")
 (item "How to make a language"))


(slide
 #:title "What is Racket"
 (item "Racket is a Lisp")
 (item "Racket is an academic language")
 (item "Multi-paradigm")
 (item "Typed and Untyped")
 (item "Bundled with an editor"))

(slide
 (titlet "Mundane interesting features"))

(full-source-slide
 #:title "Module System"
 #<<source
#lang racket
(module+ test
  (require rackunit)
  (define ε 1e-10))
 
(provide to-energy)
 
(define (to-energy m)
  (* m (expt 299792458.0 2)))
 
(module+ test
  (check-= (to-energy 0) 0 ε)
  (check-= (to-energy 1) 9e+16 1e+15))

(define pi 3.14592)

source
 )

(full-source-slide
 #:title "OOP"
#<<source
#lang racket
(define fish%
  (class object%
    (init size)
 
    (define current-size size)
 
    (super-new)
 
    (define/public (get-size)
      current-size)
 
    (define/public (grow amt)
      (set! current-size (+ amt current-size)))
 
    (define/public (eat other-fish)
      (grow (send other-fish get-size)))))
source
 )



(full-source-slide
 #:title "First Class Classes"
#<<source
#lang racket
(define (maybe-picky-mixin %)
  (if (> (random 0 10) 0)
      %
      (class % (super-new)
        (define/override (grow amt)
          (super grow (* 3/4 amt))))))

(maybe-picky-mixin fish%)

;; made up other class
(maybe-picky-mixin language%)
source
)

(source-slide
 #:title "Pattern Matching"
 #<<source
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
source
 )


(slide
 (titlet "Advanced Feature"))

  
(full-source-slide
 #:title "Continuations"
#<<source
#lang racket
(require racket/control)

(define (authenticated-request username)
  (prompt
   (let [(password
          (begin
            (print "password:")
            (read)))
         (url (let/cc k (abort k)))]
     (println
      (format "request with creds ~a: ~a for ~a"
              username
              password
              url)))))


(define auth-for-jimmy
  (authenticated-request "jimmyhmiller"))
(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; "password:" <type-password-here>
;; "request with creds jimmyhmiller: my-password for /user"
;; "request with creds jimmyhmiller: my-password for /about"

source
 )



(slide
 #:title "Power of continuations"
 (item "Debugger")
 (item "Control Flow (Exceptions)")
 (item "Concurrency")
 (item "Time Travel"))


(slide
 #:title "Other Mundane Features"
 (point "Contracts")
 (point "Gradual Typing (better than optional)")
 (point "Macros")
 (point "Extensible Editor"))

(slide
 (titlet "A Programming-Language Programming Language")
 (bitmap (build-path (collection-path "icons") "plt.gif")))

(slide
 (titlet "#lang <your-lang-here>"))

(source-slide
 #<<source
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
source
 )


(source-slide
 #<<source
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
source
 )


(source-slide
 #:result-height 3/4
 #<<source
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
source
 )



(source-slide
 #<<source
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
source
 )

(source-slide
 #<<source
#lang c
#include <stdio.h>

int main() {
  int x = 0.0;
  x++;
  float r = x / x;
  printf("%4.2f\n", r);
  r = 1.0 / 2.0;
  printf("%4.2f\n", r);
  return 0;
}
source
 )



(source-slide
 #<<source
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
source
 )


(source-slide
 #<<source
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
source
 )


(full-source-slide
 #<<source
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
source
 )


(source-slide
 #<<source
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

source
 )


(full-source-slide
 #<<source
#lang slideshow
(slide
 (titlet "A Programming-Language Programming Language")
 (bitmap (build-path (collection-path "icons") "plt.gif"))
source
 )


(slide
 #:title "Building a Racket Language"
 (item "Write a litte library")
 (item "Write a translator")
 (item "Parse input")
 (item "Combine into macros"))


(define stacker-example-source
  #<<source
#lang reader "stacker.rkt"
4
8
+
3
*
source
  )

(define stacker-rg (make-repl-group))
(define stacker-example-backing (make-module-backing stacker-rg stacker-example-source #:module-name "stacker-example.rkt"))

(slide
 #:title "Stacker"
 (module-area stacker-example-backing
              #:width (* client-w 9/10)
              #:height (* client-h (- 1 3/10 1/10))
              #:auto-eval? #t)
 (result-area stacker-rg
              #:width (* client-w 9/10)
              #:height (* client-h 3/10)))

(source-slide
#<<source
#lang br/quicklang
(define stack empty)

(define (pop-stack!)
  (define arg (first stack))
  (set! stack (rest stack))
  arg)

(define (push-stack! arg)
  (set! stack (cons arg stack)))

(define (handle [arg #f])
  (cond
    [(number? arg) (push-stack! arg)]
    [(or (equal? * arg) (equal? + arg))
     (define op-result (arg (pop-stack!)
                            (pop-stack!))) 
     (push-stack! op-result)]))

(handle 1)
(handle 2)
(handle '+)

(display (first stack))

source
)


(full-source-slide
 #<<source
#lang br/quicklang

(define (read-syntax path port)
  (define src-lines (port->lines port))
  (define src-datums (format-datums
                      '(handle ~a) src-lines))
  (define module-datum `(module stacker-mod
                          "stacker.rkt"
                          ,@src-datums))
  (datum->syntax #f module-datum))

(provide read-syntax)

(define-macro (stacker-module-begin HANDLE-EXPR ...)
  #'(#%module-begin
     HANDLE-EXPR ...
     (display (first stack))))

(provide
 (rename-out [stacker-module-begin #%module-begin]))

source
 )




(slide
 (titlet "Not just toys"))


(source-slide
 #:title "Occurrence Typing"
 #<<source
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3)))

source
 )


(source-slide
 #:title "Dependent Types"
 #<<source
#lang typed/racket #:with-refinements
(require racket/unsafe/ops)
(: safe-ref (All (A)
                 (-> ([v : (Vectorof A)]
                      [n : Natural])
                     #:pre (v n)
                           (< n (vector-length v))
                     A)))
(define (safe-ref v n) (unsafe-vector-ref v n))
(safe-ref (vector "safe!") 0)
source
 )

(source-slide
 #:title "Impressive Types"
 #<<source
#lang typed/racket #:with-refinements
(if (< 5 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
source
 )


(slide
 #:title "How is this possible?"
 (point "Radically simple base")
 (point "Code is data")
 (point "Dedication, Hardwork, and Inventiveness")
 (point "Rigor"))

(slide
 #:title "Racket's Influence"
 (point "Rust's macro system is borrowed from Racket")
 (point "Clojure spec inspired by contract system")
 (point "Typescript's type features borrow from Racket"))

(slide
 (titlet "Why does this matter?"))

(slide
 (titlet "Languages are system for thought"))




(slide)

(slide
 (titlet "Appendix"))




(source-slide
 #:result-height 3/4
 #<<source
 #lang s-exp "stackerizer.rkt"
(* 1 2 3 4 (* 5 (+ 6 7)))
source
 )

(full-source-slide
 #<<source
#lang br/quicklang
(provide + *)

(define-macro (stackerizer-mb EXPR)
  #'(#%module-begin
     (for-each displayln (reverse (flatten EXPR)))))
(provide (rename-out [stackerizer-mb #%module-begin]))

(define-macro (define-ops OP ...)
  #'(begin
      (define-macro-cases OP
        [(OP FIRST) #'FIRST]
        [(OP FIRST NEXT (... ...))
         #'(list 'OP FIRST (OP NEXT (... ...)))]) 
      ...))

(define-ops + *)
source
 )







