;; @ Saturday, April 6th, 2019 1:36:58pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang racket
(define (fact n)
  (if (= n 0) 1 (* n (fact (sub1 n)))))
(provide fact)
;; @ Saturday, April 6th, 2019 1:37:05pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang racket
(define (fact n)
  (if (= n 0) 1 (* n (fact (sub1 n)))))
(provide fact)
;; @ Saturday, April 6th, 2019 1:37:09pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(require "fact.rkt")
(fact 5)
;; @ Saturday, April 6th, 2019 1:37:09pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang racket
(define (fact n)
  (if (= n 0) 1 (* n (fact (sub1 n)))))
(provide fact)
;; @ Saturday, April 6th, 2019 1:37:13pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(require "fact.rkt")
(fact 5)
;; @ Saturday, April 6th, 2019 1:37:13pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang racket
(define (fact n)
  (if (= n 0) 1 (* n (fact (sub1 n)))))
(provide fact)
;; @ Saturday, April 6th, 2019 1:37:15pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang racket
(define (fact n)
  (if (= n 0) 1 (* n (fact (sub1 n)))))
(provide fact)
;; @ Saturday, April 6th, 2019 1:38:33pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang racket
(define (fact n)
  (if (= n 0) 1 (* n (fact (sub1 n)))))
(provide fact)
;; @ Saturday, April 6th, 2019 1:38:39pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang racket
(define (fact n)
  (if (= n 0) 1 (* n (fact (sub1 n)))))
(provide fact)
;; @ Saturday, April 6th, 2019 1:38:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(require "fact.rkt")
(fact 5)
;; @ Saturday, April 6th, 2019 1:38:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang racket
(define (fact n)
  (if (= n 0) 1 (* n (fact (sub1 n)))))
(provide fact)
;; @ Saturday, April 6th, 2019 1:38:46pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang racket
(define (fact n)
  (if (= n 0) 1 (* n (fact (sub1 n)))))
(provide fact)
;; @ Saturday, April 6th, 2019 1:52:36pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
;; @ Saturday, April 6th, 2019 3:25:37pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:25:44pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:25:45pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:25:46pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:25:46pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:25:46pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:25:47pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:25:47pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:26:26pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:26:28pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:26:31pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:27:07pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:27:29pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:27:46pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:27:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:28:01pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:28:19pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:28:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:28:55pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:29:01pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:29:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:29:47pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:29:49pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/fact.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:34:36pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:37:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:43:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:43:52pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:43:57pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:43:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:44:45pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:44:48pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Saturday, April 6th, 2019 3:44:56pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:44:58pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Saturday, April 6th, 2019 3:45:02pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:45:05pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 3:46:43pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 4:07:52pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 4:07:57pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
source 
)


(source-slide
 #<<source
#lang javascript
var x = function () {
  return "Hello Javascript";
}

x()
;; @ Saturday, April 6th, 2019 4:08:02pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
source 
)


(source-slide
 #<<source
#lang javascript
var x = function () {
  return "Hello Javascript";
}

x()
;; @ Saturday, April 6th, 2019 4:08:04pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
source 
)


(source-slide
 #<<source
#lang javascript
var x = function () {
  return "Hello Javascript";
}

x()
;; @ Saturday, April 6th, 2019 4:08:27pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 4:08:30pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Saturday, April 6th, 2019 4:08:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}

x()
;; @ Saturday, April 6th, 2019 4:09:03pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 4:09:04pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Saturday, April 6th, 2019 4:09:06pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
"Hell World"
;; @ Saturday, April 6th, 2019 4:09:09pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
"Hell World"
;; @ Saturday, April 6th, 2019 4:09:29pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
print("Hello World")
;; @ Saturday, April 6th, 2019 4:09:37pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
"Hello World"
;; @ Saturday, April 6th, 2019 4:09:43pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
2+2
;; @ Saturday, April 6th, 2019 4:10:01pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = 2;
;; @ Saturday, April 6th, 2019 4:10:03pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = 2;
x
;; @ Saturday, April 6th, 2019 4:10:06pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = 2;
x
;; @ Saturday, April 6th, 2019 4:10:14pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = 2;
x
;; @ Saturday, April 6th, 2019 4:14:43pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Saturday, April 6th, 2019 4:14:43pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}

x()
;; @ Saturday, April 6th, 2019 4:14:47pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}

x()
;; @ Saturday, April 6th, 2019 4:15:57pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 4:16:09pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 4:16:19pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 4:17:23pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 4:21:01pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 4:21:04pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
;; @ Saturday, April 6th, 2019 4:21:04pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Saturday, April 6th, 2019 4:21:21pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
2.2
;; @ Saturday, April 6th, 2019 4:21:23pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
2.2
;; @ Saturday, April 6th, 2019 4:21:29pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
print(2)
;; @ Saturday, April 6th, 2019 4:21:37pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:21:55pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:21:56pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:21:57pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:21:57pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:21:57pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:21:58pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:21:58pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:21:58pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:21:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:21:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:21:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:21:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:21:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:22:00pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:22:00pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:22:00pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = function () {
  return "Hello Javascript";
}
x()
;; @ Saturday, April 6th, 2019 4:22:18pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = 25
;; @ Saturday, April 6th, 2019 4:22:26pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang javascript
var x = 25
;; @ Saturday, April 6th, 2019 4:35:23pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 4:35:24pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Saturday, April 6th, 2019 4:35:26pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2 responds with 200
GET /72312 responds with 404
;; @ Saturday, April 6th, 2019 4:35:31pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2 responds with 200
GET /72312 responds with 404
;; @ Saturday, April 6th, 2019 4:49:52pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 4:49:56pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Saturday, April 6th, 2019 4:49:58pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Saturday, April 6th, 2019 4:50:11pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Saturday, April 6th, 2019 5:46:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Saturday, April 6th, 2019 5:49:18pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Saturday, April 6th, 2019 11:48:02pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Saturday, April 6th, 2019 11:48:12pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Saturday, April 6th, 2019 11:48:14pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Saturday, April 6th, 2019 11:48:27pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Saturday, April 6th, 2019 11:48:34pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Sunday, April 7th, 2019 12:04:36am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Sunday, April 7th, 2019 12:04:39am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Sunday, April 7th, 2019 12:04:39am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Sunday, April 7th, 2019 12:04:40am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Sunday, April 7th, 2019 12:04:42am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Sunday, April 7th, 2019 12:04:44am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Sunday, April 7th, 2019 12:04:47am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang c
#include <stdio.h>

int main() {
  int x = 0.0;
  float r = (++x) / (++x);
  printf("%4.2f\n", r);
  r = 1.0 / 2.0;
  printf("%4.2f\n", r);
  return 0;
}
;; @ Sunday, April 7th, 2019 12:05:03am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang c
#include <stdio.h>

int main() {
  int x = 0.0;
  float r = (++x) / (++x);
  printf("%4.2f\n", r);
  r = 1.0 / 2.0;
  printf("%4.2f\n", r);
  return 0;
}
;; @ Sunday, April 7th, 2019 12:05:45am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Sunday, April 7th, 2019 12:05:49am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Sunday, April 7th, 2019 12:05:51am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Sunday, April 7th, 2019 12:05:54am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Sunday, April 7th, 2019 12:05:56am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Sunday, April 7th, 2019 12:07:10am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Sunday, April 7th, 2019 12:07:13am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Sunday, April 7th, 2019 12:07:15am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Sunday, April 7th, 2019 12:07:17am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Sunday, April 7th, 2019 12:07:20am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Sunday, April 7th, 2019 12:07:25am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
;; @ Sunday, April 7th, 2019 12:07:30am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
;; @ Sunday, April 7th, 2019 12:07:37am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> x
;; @ Sunday, April 7th, 2019 12:07:44am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> x
;; @ Friday, April 12th, 2019 9:14:59am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Friday, April 12th, 2019 9:15:16am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Friday, April 12th, 2019 9:15:25am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Friday, April 12th, 2019 9:15:37am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Friday, April 12th, 2019 9:15:37am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Friday, April 12th, 2019 9:15:41am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Friday, April 12th, 2019 9:15:55am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Friday, April 12th, 2019 9:15:57am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Friday, April 12th, 2019 9:15:59am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Friday, April 12th, 2019 9:16:38am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Friday, April 12th, 2019 9:16:39am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Friday, April 12th, 2019 9:16:41am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Friday, April 12th, 2019 9:16:56am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Friday, April 12th, 2019 9:17:02am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Friday, April 12th, 2019 9:17:09am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Friday, April 12th, 2019 9:17:21am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Friday, April 12th, 2019 9:17:24am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Sunday, April 14th, 2019 8:29:25pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Sunday, April 14th, 2019 8:29:30pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Sunday, April 14th, 2019 8:29:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Sunday, April 14th, 2019 8:29:36pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Sunday, April 14th, 2019 8:29:39pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Sunday, April 14th, 2019 8:29:43pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Sunday, April 14th, 2019 8:29:46pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Sunday, April 14th, 2019 8:30:03pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(def x : Integer
  (let ([y 3]
        [z 7])
    {y + z}))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Sunday, April 14th, 2019 8:30:22pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(def x : Integer
  (let ([y 3]
        [z 7])
    {y + z}))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Sunday, April 14th, 2019 8:31:09pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Sunday, April 14th, 2019 8:31:09pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Sunday, April 14th, 2019 8:31:12pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Sunday, April 14th, 2019 8:31:13pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Sunday, April 14th, 2019 8:31:15pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Sunday, April 14th, 2019 8:31:16pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Sunday, April 14th, 2019 8:31:20pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])

(show (Just 1))

;; @ Sunday, April 14th, 2019 8:31:33pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])

(show (Just 1))

;; @ Sunday, April 14th, 2019 8:31:49pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Sunday, April 14th, 2019 8:31:55pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Sunday, April 14th, 2019 8:31:57pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Sunday, April 14th, 2019 8:31:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Sunday, April 14th, 2019 8:32:02pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Sunday, April 14th, 2019 8:32:06pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Sunday, April 14th, 2019 8:32:07pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Sunday, April 14th, 2019 8:32:09pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])

(show (Just Nothing))

;; @ Sunday, April 14th, 2019 8:32:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Sunday, April 14th, 2019 8:32:44pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Sunday, April 14th, 2019 8:32:45pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Sunday, April 14th, 2019 8:32:45pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Sunday, April 14th, 2019 8:32:46pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Sunday, April 14th, 2019 8:32:47pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Sunday, April 14th, 2019 8:32:47pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Sunday, April 14th, 2019 8:32:48pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])

(show Nothing)

;; @ Sunday, April 14th, 2019 8:34:17pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])

(instance (Show Integer)
  [show (λ* "thing")])

(show 1)

;; @ Sunday, April 14th, 2019 8:34:31pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])

(instance (Show Integer)
  [show (λ (_) "thing")])

(show 1)

;; @ Sunday, April 14th, 2019 8:34:58pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])

(instance (Show Integer)
  [show (λ (_) "thing")])

(show (Just 1))

;; @ Sunday, April 14th, 2019 8:35:05pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])

(instance (Show Integer)
  [show (λ (_) "thing")])

(show (Just 1))

;; @ Sunday, April 14th, 2019 8:50:34pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Sunday, April 14th, 2019 8:50:35pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Sunday, April 14th, 2019 8:50:36pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Sunday, April 14th, 2019 8:50:38pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Sunday, April 14th, 2019 8:50:38pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Sunday, April 14th, 2019 8:50:39pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Sunday, April 14th, 2019 8:50:41pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Sunday, April 14th, 2019 8:50:42pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Sunday, April 14th, 2019 8:50:44pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Sunday, April 14th, 2019 8:51:00pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Sunday, April 14th, 2019 8:51:01pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Sunday, April 14th, 2019 8:51:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Sunday, April 14th, 2019 8:55:45pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Sunday, April 14th, 2019 8:55:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Sunday, April 14th, 2019 8:55:53pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Sunday, April 14th, 2019 8:55:53pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Sunday, April 14th, 2019 8:55:54pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Sunday, April 14th, 2019 8:55:54pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Sunday, April 14th, 2019 8:55:54pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Sunday, April 14th, 2019 8:56:00pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).


ancestor(A, douglas)?

;; @ Sunday, April 14th, 2019 8:58:00pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).


ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 6:52:52pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 6:52:53pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 6:52:56pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 6:52:56pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Monday, April 15th, 2019 6:52:56pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Monday, April 15th, 2019 6:52:56pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 6:52:56pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Monday, April 15th, 2019 6:52:56pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Monday, April 15th, 2019 6:54:07pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 6:54:10pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 6:54:13pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 6:54:13pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 6:54:13pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 6:54:13pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Monday, April 15th, 2019 6:54:14pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Monday, April 15th, 2019 6:54:14pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Monday, April 15th, 2019 6:54:15pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Monday, April 15th, 2019 6:55:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 6:55:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Monday, April 15th, 2019 6:55:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Monday, April 15th, 2019 6:55:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Monday, April 15th, 2019 6:55:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 6:55:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Monday, April 15th, 2019 6:55:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 6:57:15pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 6:57:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 6:57:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 6:57:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 6:57:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Monday, April 15th, 2019 6:57:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Monday, April 15th, 2019 6:57:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 6:57:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Monday, April 15th, 2019 6:57:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 6:57:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Monday, April 15th, 2019 6:57:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 7:02:30pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 7:08:33pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 7:11:29pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 7:40:15pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 8:20:24pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 8:31:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
(define x (read))
x
;; @ Monday, April 15th, 2019 8:31:17pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
(define x (read))
x
;; @ Monday, April 15th, 2019 9:09:23pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define users-table (make-hash))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:09:47pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define users-table (make-hash))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:10:55pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define users-table (make-hash))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:11:20pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 9:11:20pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define users-table (make-hash))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:11:30pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 9:11:34pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 9:14:24pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define users-table (make-hash))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:14:39pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define users-table (make-hash))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:15:47pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define users-table (make-hash))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") "password")))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:16:04pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define users-table (make-hash))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") "password"))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:16:18pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define users-table (make-hash))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (println "password:") "password"))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:16:57pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define users-table (make-hash))

(define (authenticated-request username)
  (prompt
   (let [(password
          (begin (println "password:")
                 (read)))
         (url (let/cc k (abort k)))]
     (println
      (format
       "request with creds ~a: ~a for ~a"
       username
       password
       url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:18:04pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define-values (in out) (make-pipe))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read in)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:18:29pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define-values (in out) (make-pipe))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read in)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:18:33pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define-values (in out) (make-pipe))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read in)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:18:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define-values (in out) (make-pipe))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read in)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:33:33pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define-values (in out) (make-pipe))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read in)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:35:57pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define-values (in out) (make-pipe))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read in)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:36:10pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define-values (in out) (make-pipe))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read in)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:36:33pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define-values (in out) (make-pipe))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read in)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:36:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define-values (in out) (make-pipe))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read in)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:40:30pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define-values (in out) (make-pipe))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read in)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))


(define auth-for-jimmy (authenticated-request "jimmyhmiller"))

(auth-for-jimmy "/user")
(auth-for-jimmy "/about")

;; @ Monday, April 15th, 2019 9:42:22pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define-values (in out) (make-pipe))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read in)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))

(define (run)
  (define auth-for-jimmy (authenticated-request "jimmyhmiller"))

  (auth-for-jimmy "/user")
  (auth-for-jimmy "/about"))

(provide in out run)

;; @ Monday, April 15th, 2019 9:43:55pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define-values (in out) (make-pipe))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read in)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))

(define (run)
  (define auth-for-jimmy (authenticated-request "jimmyhmiller"))

  (auth-for-jimmy "/user")
  (auth-for-jimmy "/about"))

(provide in out run)

;; @ Monday, April 15th, 2019 9:45:06pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket

(require racket/control)

(define-values (in out) (make-pipe))

(define (authenticated-request username)
  (prompt
   (let [(password (begin (print "password:") (read in)))
         (url (let/cc k (abort k)))]
     (println (format "request with creds ~a: ~a for ~a" username password url)))))

(define (run)
  (define auth-for-jimmy (authenticated-request "jimmyhmiller"))
  (begin
    (auth-for-jimmy "/user")
    (auth-for-jimmy "/about")))

(provide in out run)

;; @ Monday, April 15th, 2019 9:48:21pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 9:51:37pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 9:51:47pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 9:51:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 9:51:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 9:54:05pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 9:54:06pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 9:54:07pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 9:54:16pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 9:55:31pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 9:55:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 9:55:34pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 9:55:43pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 9:56:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 9:56:09pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 9:56:10pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 9:56:18pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 9:56:21pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 9:56:36pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 9:56:37pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 9:56:47pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 9:56:51pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 10:29:30pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 10:29:30pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 10:29:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Monday, April 15th, 2019 10:29:33pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Monday, April 15th, 2019 10:29:33pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Monday, April 15th, 2019 10:29:34pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 10:29:35pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang slideshow
(slide
 (titlet "A Programming-Language Programming Language"))
;; @ Monday, April 15th, 2019 10:29:39pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang slideshow
(slide
 (titlet "A Programming-Language Programming Language"))
;; @ Monday, April 15th, 2019 10:30:14pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Monday, April 15th, 2019 10:30:14pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Monday, April 15th, 2019 10:30:14pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 10:30:14pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 10:30:14pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Monday, April 15th, 2019 10:30:15pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Monday, April 15th, 2019 10:32:10pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 10:32:11pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 10:32:12pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 10:32:12pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 10:32:13pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Monday, April 15th, 2019 10:32:14pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Monday, April 15th, 2019 10:32:15pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Monday, April 15th, 2019 10:32:16pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 10:33:36pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 10:33:43pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 10:33:45pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 10:33:46pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Monday, April 15th, 2019 10:33:48pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Monday, April 15th, 2019 10:33:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Monday, April 15th, 2019 10:33:53pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Monday, April 15th, 2019 10:33:56pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 10:33:58pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 10:35:22pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 10:35:31pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 10:35:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 10:35:34pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 10:37:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 10:37:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 10:38:00pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 10:38:06pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Monday, April 15th, 2019 10:38:07pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 10:38:07pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Monday, April 15th, 2019 10:38:07pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Monday, April 15th, 2019 10:38:07pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Monday, April 15th, 2019 10:38:33pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 10:38:34pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 10:38:35pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 10:38:35pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 10:38:37pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Monday, April 15th, 2019 10:38:37pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Monday, April 15th, 2019 10:38:38pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Monday, April 15th, 2019 10:38:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 10:40:22pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Monday, April 15th, 2019 10:40:22pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Monday, April 15th, 2019 10:40:22pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 10:40:22pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 10:40:22pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 10:40:22pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Monday, April 15th, 2019 10:40:22pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Monday, April 15th, 2019 10:40:22pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 10:40:22pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Monday, April 15th, 2019 10:40:55pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Monday, April 15th, 2019 10:40:55pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Monday, April 15th, 2019 10:40:55pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Monday, April 15th, 2019 10:40:56pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Monday, April 15th, 2019 10:40:57pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Monday, April 15th, 2019 10:40:57pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Monday, April 15th, 2019 10:40:57pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Monday, April 15th, 2019 10:54:44pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 (define (magnitude n)
   (match n
     [(list’ cart (? real xs) ...)
      (sqrt (apply + (map sqr xs)))]
     
     [(list ’polar (? real? r) (? real? theta) ...)
      r]))
;; @ Monday, April 15th, 2019 10:54:44pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 (define (magnitude n)
   (match n
     [(list’ cart (? real xs) ...)
      (sqrt (apply + (map sqr xs)))]
     
     [(list ’polar (? real? r) (? real? theta) ...)
      r]))
;; @ Monday, April 15th, 2019 10:54:51pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 (define (magnitude n)
   (match n
     [(list’ cart (? real xs) ...)
      (sqrt (apply + (map sqr xs)))]
     
     [(list ’polar (? real? r) (? real? theta) ...)
      r]))
;; @ Monday, April 15th, 2019 10:55:20pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#racket
(define (magnitude n)
  (match n
    [(list’ cart (? real xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))
;; @ Monday, April 15th, 2019 10:55:24pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#racket
(define (magnitude n)
  (match n
    [(list’ cart (? real xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))
;; @ Monday, April 15th, 2019 10:55:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list’ cart (? real xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))
;; @ Monday, April 15th, 2019 10:55:44pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list’ cart (? real xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))
;; @ Monday, April 15th, 2019 10:56:00pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))
;; @ Monday, April 15th, 2019 10:56:07pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))
;; @ Monday, April 15th, 2019 10:56:31pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Monday, April 15th, 2019 10:56:38pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Monday, April 15th, 2019 11:27:56pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/stacker-example.rkt
#lang reader "stacker.rkt"
4
8
+
3
*
;; @ Monday, April 15th, 2019 11:28:15pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/stacker-example.rkt
#lang reader "stacker.rkt"
4
8
+
3
*
;; @ Monday, April 15th, 2019 11:29:26pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/stacker-example.rkt
#lang reader "stacker.rkt"
4
8
+
3
*
;; @ Monday, April 15th, 2019 11:29:43pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/stacker-example.rkt
#lang reader "stacker.rkt"
4
8
+
3
*
;; @ Monday, April 15th, 2019 11:38:35pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
     (define op-result (arg (pop-stack!) (pop-stack!))) 
     (push-stack! op-result)]))

(handle 1)
(handle 2)
(handle '+)

(display (first stack))

;; @ Monday, April 15th, 2019 11:38:53pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
     (define op-result (arg (pop-stack!) (pop-stack!))) 
     (push-stack! op-result)]))

(handle 1)
(handle 2)
(handle '+)

(display stack)

;; @ Monday, April 15th, 2019 11:39:02pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
     (define op-result (arg (pop-stack!) (pop-stack!))) 
     (push-stack! op-result)]))

(handle 1)
(handle 2)
(handle +)

(display stack)

;; @ Monday, April 15th, 2019 11:39:11pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
     (define op-result (arg (pop-stack!) (pop-stack!))) 
     (push-stack! op-result)]))

(handle 1)
(handle 2)
(handle +)

(display (first stack))

;; @ Monday, April 15th, 2019 11:39:26pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
(handle +)

(display (first stack))

;; @ Monday, April 15th, 2019 11:51:58pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 #lang s-exp "stackerizer.rkt"
(* 1 2 (+ 3 4 (* 5 6 (+ 7 8 (* 9 10)))))
;; @ Monday, April 15th, 2019 11:52:06pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 #lang s-exp "stackerizer.rkt"
(* 1 2 (+ 3 4 (* 5 6 (+ 7 8 (* 9 10)))))
;; @ Monday, April 15th, 2019 11:52:31pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 #lang s-exp "stackerizer.rkt"
(* 1 2 (+ 3 4 (* 5 6 (+ 7 8 (* 9 10)))))
;; @ Monday, April 15th, 2019 11:52:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 #lang s-exp "stackerizer.rkt"
(* 1 2 (+ 3 4 (* 5 6 (+ 7 8 (* 9 10)))))
;; @ Monday, April 15th, 2019 11:53:29pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 #lang s-exp "stackerizer.rkt"
(* 1 2 3 4 (* 5 6 (+ 7 8 9))))
;; @ Monday, April 15th, 2019 11:53:33pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 #lang s-exp "stackerizer.rkt"
(* 1 2 3 4 (* 5 6 (+ 7 8 9))))
;; @ Monday, April 15th, 2019 11:53:45pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 #lang s-exp "stackerizer.rkt"
(* 1 2 3 4 (* 5 6 (+ 7 8 9)))
;; @ Monday, April 15th, 2019 11:54:16pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 #lang s-exp "stackerizer.rkt"
(* 1 2 3 4 (* 5 6 (+ 7 8)))
;; @ Monday, April 15th, 2019 11:54:38pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 #lang s-exp "stackerizer.rkt"
(* 1 2 3 4 (* 5 (+ 6 7)))
;; @ Monday, April 15th, 2019 11:54:54pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 #lang s-exp "stackerizer.rkt"
(* 1 2 3 4 (* 5 (+ 6 7)))
;; @ Tuesday, April 16th, 2019 12:58:16am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 12:58:29am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Tuesday, April 16th, 2019 12:58:33am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Tuesday, April 16th, 2019 12:59:43am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 12:59:48am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:01:04am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:01:09am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:01:11am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:01:13am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:01:15am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:01:17am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:01:19am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:01:22am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:01:24am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:01:28am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:02:17am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:02:36am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Tuesday, April 16th, 2019 1:02:40am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Tuesday, April 16th, 2019 1:02:42am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Tuesday, April 16th, 2019 1:02:43am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Tuesday, April 16th, 2019 1:02:43am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Tuesday, April 16th, 2019 1:02:44am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Tuesday, April 16th, 2019 1:02:45am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Tuesday, April 16th, 2019 1:02:46am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Tuesday, April 16th, 2019 1:02:49am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:02:52am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:03:34am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:03:45am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Tuesday, April 16th, 2019 1:03:50am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Tuesday, April 16th, 2019 1:03:51am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Tuesday, April 16th, 2019 1:03:52am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Tuesday, April 16th, 2019 1:03:53am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Tuesday, April 16th, 2019 1:03:57am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Tuesday, April 16th, 2019 1:03:58am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 1:03:59am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Tuesday, April 16th, 2019 1:04:00am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Tuesday, April 16th, 2019 1:04:01am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Tuesday, April 16th, 2019 1:04:07am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang hackett

(data (Maybe a)
  Nothing
  (Just a))

(class (Show a)
  [show : {a -> String}])

(instance (forall [a] (Show a) => (Show (Maybe a)))
  [show (λ* [[(Just x)] {"(Just " ++ (show x) ++ ")"}]
            [[Nothing ] "Nothing"])])
;; @ Tuesday, April 16th, 2019 1:04:27am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Tuesday, April 16th, 2019 1:04:28am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 1:04:29am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Tuesday, April 16th, 2019 1:04:29am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Tuesday, April 16th, 2019 1:04:33am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 1:04:33am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Tuesday, April 16th, 2019 1:04:34am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Tuesday, April 16th, 2019 1:04:35am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Tuesday, April 16th, 2019 1:04:36am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Tuesday, April 16th, 2019 1:04:39am -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Tuesday, April 16th, 2019 1:04:47am -----------
;; /Users/jimmyhmiller/Desktop/scheme/stacker-example.rkt
#lang reader "stacker.rkt"
4
8
+
3
*
;; @ Tuesday, April 16th, 2019 3:16:09pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
(: flexible-length (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))
;; @ Tuesday, April 16th, 2019 3:16:12pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
(: flexible-length (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))
;; @ Tuesday, April 16th, 2019 3:16:41pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))
;; @ Tuesday, April 16th, 2019 3:16:53pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))
;; @ Tuesday, April 16th, 2019 3:17:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3)))

;; @ Tuesday, April 16th, 2019 3:18:04pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3)))

;; @ Tuesday, April 16th, 2019 3:24:12pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(require racket/unsafe/ops)
(: safe-ref (All (A)
                 (-> ([v : (Vectorof A)]
                      [n : Natural])
                     #:pre (v n) (< n (vector-length v))
                     A)))
(define (safe-ref v n) (unsafe-vector-ref v n))
(safe-ref (vector "safe!") 0)
;; @ Tuesday, April 16th, 2019 3:24:19pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(require racket/unsafe/ops)
(: safe-ref (All (A)
                 (-> ([v : (Vectorof A)]
                      [n : Natural])
                     #:pre (v n) (< n (vector-length v))
                     A)))
(define (safe-ref v n) (unsafe-vector-ref v n))
(safe-ref (vector "safe!") 0)
;; @ Tuesday, April 16th, 2019 3:25:26pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(require racket/unsafe/ops)
(: safe-ref (All (A)
                 (-> ([v : (Vectorof A)]
                      [n : Natural])
                     #:pre (v n) (< n (vector-length v))
                     A)))
(define (safe-ref v n) (unsafe-vector-ref v n))
(safe-ref (vector "safe!") 0)
;; @ Tuesday, April 16th, 2019 3:25:28pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(require racket/unsafe/ops)
(: safe-ref (All (A)
                 (-> ([v : (Vectorof A)]
                      [n : Natural])
                     #:pre (v n) (< n (vector-length v))
                     A)))
(define (safe-ref v n) (unsafe-vector-ref v n))
(safe-ref (vector "safe!") 0)
;; @ Tuesday, April 16th, 2019 3:25:35pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(require racket/unsafe/ops)
(: safe-ref (All (A)
                 (-> ([v : (Vectorof A)]
                      [n : Natural])
                     #:pre (v n) (< n (vector-length v))
                     A)))
(define (safe-ref v n) (unsafe-vector-ref v n))
(safe-ref (vector "safe!") 0)
;; @ Tuesday, April 16th, 2019 3:25:37pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(require racket/unsafe/ops)
(: safe-ref (All (A)
                 (-> ([v : (Vectorof A)]
                      [n : Natural])
                     #:pre (v n) (< n (vector-length v))
                     A)))
(define (safe-ref v n) (unsafe-vector-ref v n))
(safe-ref (vector "safe!") 0)
;; @ Tuesday, April 16th, 2019 3:25:45pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(require racket/unsafe/ops)
(: safe-ref (All (A)
                 (-> ([v : (Vectorof A)]
                      [n : Natural])
                     #:pre (v n) (< n (vector-length v))
                     A)))
(define (safe-ref v n) (unsafe-vector-ref v n))
(safe-ref (vector "safe!") 0)
;; @ Tuesday, April 16th, 2019 3:25:47pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(require racket/unsafe/ops)
(: safe-ref (All (A)
                 (-> ([v : (Vectorof A)]
                      [n : Natural])
                     #:pre (v n) (< n (vector-length v))
                     A)))
(define (safe-ref v n) (unsafe-vector-ref v n))
(safe-ref (vector "safe!") 0)
;; @ Tuesday, April 16th, 2019 3:25:58pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(require racket/unsafe/ops)
(: safe-ref (All (A)
                 (-> ([v : (Vectorof A)]
                      [n : Natural])
                     #:pre (v n) (< n (vector-length v))
                     A)))
(define (safe-ref v n) (unsafe-vector-ref v n))
(safe-ref (vector "safe!") 0)
;; @ Tuesday, April 16th, 2019 3:26:06pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(require racket/unsafe/ops)
(: safe-ref (All (A)
                 (-> ([v : (Vectorof A)]
                      [n : Natural])
                     #:pre (v n) (< n (vector-length v))
                     A)))
(define (safe-ref v n) (unsafe-vector-ref v n))
(safe-ref (vector "safe!") 0)
;; @ Tuesday, April 16th, 2019 3:26:41pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 3:26:51pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(require racket/unsafe/ops)
(: safe-ref (All (A)
                 (-> ([v : (Vectorof A)]
                      [n : Natural])
                     #:pre (v n)
                           (< n (vector-length v))
                     A)))
(define (safe-ref v n) (unsafe-vector-ref v n))
(safe-ref (vector "safe!") 1)
;; @ Tuesday, April 16th, 2019 3:27:01pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(if (< 5 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 3:27:07pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 3:27:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3)))

;; @ Tuesday, April 16th, 2019 3:27:34pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 3:27:39pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(if (< 5 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 3:27:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 3:27:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3)))

;; @ Tuesday, April 16th, 2019 3:27:47pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 #lang s-exp "stackerizer.rkt"
(* 1 2 3 4 (* 5 (+ 6 7)))
;; @ Tuesday, April 16th, 2019 3:27:54pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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

;; @ Tuesday, April 16th, 2019 3:27:56pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/stacker-example.rkt
#lang reader "stacker.rkt"
4
8
+
3
*
;; @ Tuesday, April 16th, 2019 3:28:01pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Tuesday, April 16th, 2019 3:28:07pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Tuesday, April 16th, 2019 3:28:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Tuesday, April 16th, 2019 3:28:09pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 3:28:10pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Tuesday, April 16th, 2019 3:28:10pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Tuesday, April 16th, 2019 3:28:11pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Tuesday, April 16th, 2019 3:28:11pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Tuesday, April 16th, 2019 3:28:15pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 3:29:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(if (< 5 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 3:29:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/stacker-example.rkt
#lang reader "stacker.rkt"
4
8
+
3
*
;; @ Tuesday, April 16th, 2019 3:29:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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

;; @ Tuesday, April 16th, 2019 3:29:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3)))

;; @ Tuesday, April 16th, 2019 3:29:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 3:29:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 #lang s-exp "stackerizer.rkt"
(* 1 2 3 4 (* 5 (+ 6 7)))
;; @ Tuesday, April 16th, 2019 3:29:13pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/stacker-example.rkt
#lang reader "stacker.rkt"
4
8
+
3
*
;; @ Tuesday, April 16th, 2019 3:29:15pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/stacker-example.rkt
#lang reader "stacker.rkt"
4
8
+
3
*
;; @ Tuesday, April 16th, 2019 3:31:07pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(module+ test
  (require rackunit)
  (define ε 1e-10))
 
(provide to-energy)
 
(define (to-energy m)
  (* m (expt 299792458.0 2)))
 
(module+ test
  (check-= (to-energy 0) 0 ε)
  (check-= (to-energy 1) 9e+16 1e+15)))

(define pi 3.14592)

;; @ Tuesday, April 16th, 2019 3:31:11pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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

;; @ Tuesday, April 16th, 2019 3:31:49pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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

;; @ Tuesday, April 16th, 2019 3:32:06pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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

(to-energy pi)

;; @ Tuesday, April 16th, 2019 3:32:09pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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

(to-energy pi)

;; @ Tuesday, April 16th, 2019 3:32:12pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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

(to-energy pi)

;; @ Tuesday, April 16th, 2019 3:32:13pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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

(to-energy pi)

;; @ Tuesday, April 16th, 2019 3:32:26pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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


;; @ Tuesday, April 16th, 2019 3:32:34pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(module+ test
  (require rackunit)
  (define ε 1e-10))
 
(provide to-energy)
 
(define (to-energy m)
  (* m (expt 299792458.0 2)))
 
(module+ test
  (check-= (to-energy 0) 10 ε)
  (check-= (to-energy 1) 9e+16 1e+15))

(define pi 3.14592)


;; @ Tuesday, April 16th, 2019 3:33:13pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 3:44:42pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 3:44:51pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3)))

;; @ Tuesday, April 16th, 2019 3:44:51pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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

;; @ Tuesday, April 16th, 2019 3:44:51pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/stacker-example.rkt
#lang reader "stacker.rkt"
4
8
+
3
*
;; @ Tuesday, April 16th, 2019 3:44:51pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 3:44:51pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Tuesday, April 16th, 2019 3:44:52pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Tuesday, April 16th, 2019 3:44:52pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Tuesday, April 16th, 2019 3:44:52pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Tuesday, April 16th, 2019 3:44:52pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Tuesday, April 16th, 2019 3:44:52pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Tuesday, April 16th, 2019 3:44:52pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Tuesday, April 16th, 2019 3:44:52pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(if (< 5 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 3:44:52pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 3:44:56pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 #lang s-exp "stackerizer.rkt"
(* 1 2 3 4 (* 5 (+ 6 7)))
;; @ Tuesday, April 16th, 2019 3:44:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 3:45:00pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(if (< 5 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 3:53:25pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 3:59:36pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 3:59:39pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Tuesday, April 16th, 2019 3:59:39pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 3:59:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Tuesday, April 16th, 2019 3:59:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Tuesday, April 16th, 2019 3:59:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Tuesday, April 16th, 2019 3:59:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Tuesday, April 16th, 2019 3:59:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Tuesday, April 16th, 2019 3:59:44pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang htdp/bsl
;; Any key inflates the balloon

(require 2htdp/image)
(require 2htdp/universe)

(define (balloon b) (circle b "solid" "red"))

(define (blow-up b k) (+ b 5))

(define (deflate b) (max (- b 1) 1))

(big-bang 50 (on-key blow-up) (on-tick deflate)
          (to-draw balloon 200 200))
;; @ Tuesday, April 16th, 2019 3:59:52pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang htdp/bsl
;; Any key inflates the balloon

(require 2htdp/image)
(require 2htdp/universe)

(define (balloon b) (circle b "solid" "red"))

(define (blow-up b k) (+ b 5))

(define (deflate b) (max (- b 1) 1))

(big-bang 50 (on-key blow-up) (on-tick deflate)
          (to-draw balloon 200 200))
;; @ Tuesday, April 16th, 2019 4:00:06pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang htdp/bsl
;; Any key inflates the balloon

(require 2htdp/image)
(require 2htdp/universe)

(define (balloon b) (circle b "solid" "red"))

(define (blow-up b k) (+ b 5))

(define (deflate b) (max (- b 1) 1))

(big-bang 50 (on-key blow-up) (on-tick deflate)
          (to-draw balloon 200 200))
;; @ Tuesday, April 16th, 2019 4:19:18pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 4:20:15pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 4:20:38pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Tuesday, April 16th, 2019 4:20:53pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Tuesday, April 16th, 2019 4:20:53pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Tuesday, April 16th, 2019 4:20:54pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Tuesday, April 16th, 2019 4:21:02pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Tuesday, April 16th, 2019 4:21:15pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Tuesday, April 16th, 2019 4:21:18pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 4:21:23pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Tuesday, April 16th, 2019 4:21:24pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Tuesday, April 16th, 2019 4:21:29pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Tuesday, April 16th, 2019 4:21:39pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/stacker-example.rkt
#lang reader "stacker.rkt"
4
8
+
3
*
;; @ Tuesday, April 16th, 2019 4:21:43pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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

;; @ Tuesday, April 16th, 2019 4:21:58pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3)))

;; @ Tuesday, April 16th, 2019 4:22:00pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 4:22:03pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3)))

;; @ Tuesday, April 16th, 2019 4:22:07pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 4:22:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(if (< 5 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 7:00:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 7:04:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 7:06:07pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 "5"))
;; @ Tuesday, April 16th, 2019 7:06:19pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 7:08:17pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs))
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 7:17:26pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang racket
(define (magnitude n)
  (match n
    [(list 'cart (? real? xs) ...)
     (sqrt (apply + (map sqr xs)))]
     
    [(list ’polar (? real? r) (? real? theta) ...)
     r]))

(magnitude '(cart 1 2 3 4 5))
;; @ Tuesday, April 16th, 2019 7:17:50pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(struct pt ([x : Real] [y : Real]))
 
(: distance (-> pt pt Real))
(define (distance p1 p2)
  (sqrt (+ (sqr (- (pt-x p2) (pt-x p1)))
           (sqr (- (pt-y p2) (pt-y p1))))))

(distance (pt 1 2) (pt 3 5))
;; @ Tuesday, April 16th, 2019 7:18:31pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang planet dyoo/bf
++++++[>++++++++++++<-]>.
>++++++++++[>++++++++++<-]>+.
+++++++..+++.>++++[>+++++++++++<-]>.
<+++[>----<-]>.<<<<<+++[>+++++<-]>.
>>.+++.------.--------.>>+.
;; @ Tuesday, April 16th, 2019 7:19:03pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang riposte
%base := https://persist.jimmyhmiller.now.sh/
GET /2
;; @ Tuesday, April 16th, 2019 7:19:49pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang ecmascript

var x = function () {
  return "Hello World"
}

x()
;; @ Tuesday, April 16th, 2019 7:20:02pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 7:20:19pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang semilit racket
This is not part of the program
> (define x 1) ;; part of the program
> 1
;; @ Tuesday, April 16th, 2019 7:21:28pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang basic
30 rem print 'ignored'
35
50 print "never gets here"
40 end
60 print 'three' : print 1.0 + 3
70 goto 11. + 18.5 + .5 rem ignored
10 print "o" ; "n" ; "e"
20 print : goto 60.0 : end
;; @ Tuesday, April 16th, 2019 7:23:27pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, douglas)?

;; @ Tuesday, April 16th, 2019 7:24:29pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(A, B)?

;; @ Tuesday, April 16th, 2019 7:24:36pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang datalog
parent(john, douglas).
parent(bob, john).
parent(ebbon, bob).

ancestor(A, B) :- parent(A, B).
ancestor(A, B) :-
    parent(A, C),
    ancestor(C, B).

ancestor(ebbon, B)?

;; @ Tuesday, April 16th, 2019 7:27:16pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/stacker-example.rkt
#lang reader "stacker.rkt"
4
8
+
3
*
;; @ Tuesday, April 16th, 2019 7:27:57pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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

;; @ Tuesday, April 16th, 2019 7:30:26pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
(handle +)

(display (first stack))

;; @ Tuesday, April 16th, 2019 7:34:13pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3)))

;; @ Tuesday, April 16th, 2019 7:36:46pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length 2))

;; @ Tuesday, April 16th, 2019 7:37:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (display "thing")
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3))

;; @ Tuesday, April 16th, 2019 7:37:35pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (display "thing")
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3)))

;; @ Tuesday, April 16th, 2019 7:37:46pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (display "thing")
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3)))

;; @ Tuesday, April 16th, 2019 7:37:58pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (display "thing")
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3)))

;; @ Tuesday, April 16th, 2019 7:38:02pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (display "thing")
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length '(1 2 3)))

;; @ Tuesday, April 16th, 2019 7:38:17pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket
(: flexible-length
   (-> (U String (Listof Any)) Integer))
(define (flexible-length str-or-lst)
  (display "thing")
  (if (string? str-or-lst)
      (string-length str-or-lst)
      (length str-or-lst)))

(list (flexible-length "test")
      (flexible-length 1))

;; @ Tuesday, April 16th, 2019 7:38:45pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 7:40:54pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(require racket/unsafe/ops)
(: safe-ref (All (A)
                 (-> ([v : (Vectorof A)]
                      [n : Natural])
                     #:pre (v n)
                           (< n (vector-length v))
                     A)))
(define (safe-ref v n) (unsafe-vector-ref v n))
(safe-ref (vector "safe!") 1)
;; @ Tuesday, April 16th, 2019 7:42:42pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
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
;; @ Tuesday, April 16th, 2019 7:42:53pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(if (< 5 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 7:43:23pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(if (> 5 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 7:43:32pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(if (< 5 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 7:44:59pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(if (< (read) 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 7:45:08pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(if (< (real (read)) 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 7:45:28pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements

(if (< (random 0 10) 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 7:45:48pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(define x 2
(if (< x 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 7:45:52pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(define x 2)
(if (< x 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 7:46:25pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(define x 5)
(if (< x 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 7:46:40pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(define x 5)
(set! x 2)
(if (< x 4)
    (+ "Luke," "I am your father")
    "that's impossible!")
;; @ Tuesday, April 16th, 2019 7:47:00pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(define x 5)

(if (< x 4)
    (+ "Luke," "I am your father")
    "that's impossible!")

(set! x 2)
;; @ Tuesday, April 16th, 2019 7:47:15pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(define x 5)

(if (< x 4)
    (+ "Luke," "I am your father")
    "that's impossible!")

(provide x)
;; @ Tuesday, April 16th, 2019 7:47:30pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(define x 5)

(if (< x 4)
    (+ "Luke," "I am your father")
    "that's impossible!")

(set! x 2)
;; @ Tuesday, April 16th, 2019 7:47:57pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(define x 5)

(if (< x 4)
    (+ "Luke," "I am your father")
    "that's impossible!")

(eval "set! x 2")
;; @ Tuesday, April 16th, 2019 7:48:06pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(define x 5)

(if (< x 4)
    (+ "Luke," "I am your father")
    "that's impossible!")

(eval "(set! x 2)")
;; @ Tuesday, April 16th, 2019 7:48:13pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(define x 5)

(eval "(set! x 2)")
(if (< x 4)
    (+ "Luke," "I am your father")
    "that's impossible!")


;; @ Tuesday, April 16th, 2019 7:48:21pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
#lang typed/racket #:with-refinements
(define x 5)

(eval '(set! x 2))
(if (< x 4)
    (+ "Luke," "I am your father")
    "that's impossible!")


;; @ Tuesday, April 16th, 2019 7:56:03pm -----------
;; /Users/jimmyhmiller/Desktop/scheme/program.rkt
 #lang s-exp "stackerizer.rkt"
(* 1 2 3 4 (* 5 (+ 6 7)))
