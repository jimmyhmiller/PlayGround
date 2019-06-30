# Introduction to Term Rewriting with Meander

Meander is heavily inspired by the capabilities of term rewriting languages. But sadly, there aren't many introductions to term rewriting aimed at every day software engineers. Typically introductions to term rewriting immediately dive into discussing mathematical properties or proving theorems. These can be really interesting and useful in their own right. But personally I like to get an intuitive feel for something before diving into a formalism. That is the aim of this post, to help you have a more intuitive understanding of how Term Rewriting works and what it is capable of.

## The Basics

The goal of Term Rewriting is to take some bit of data and rewrite it into some other bit of data. We accomplish this by writing rules that tell us for a given piece of data what we should turn it into. 

```clojure
(require '[meander.strategy.delta :as r])

(def x-to-y
  (r/rewrite
   :x :y))

(x-to-y :x)
;; => :y
```

Here is the most simple rewrite rule imaginable. If we are given `:x` we turn it into `:y`. In term rewriting, the pattern we are using to match is often called the `left-hand-side` and the data we return is called the `right-hand-side`. So `:x` is our left-hand-side and `:y` is our right-hand-side. The data we pass in to transform is called the reducible-expression (or `redex` for short).

Admittedly, this seems almost useless, and it really is with this overly simplisitic example. But let's take it slow and build it up. 

```clojure
(def rewrite-some-keywords
  (r/rewrite
   :x :y
   :l :q
   :r :t
   :a :c))

(rewrite-some-keywords :a)
;; => :c
```

Here we've extended our rewrite to have multiple rules. Now we can handle more than just `:x`. Of course this is still really limiting. We definitely can't list every single possible input for all of our rules. We need a way to match any input. That is is where `logic-variables` come in. (Use of logic variables in Meader does deviate a bit from normal term rewriting, but we aren't going to worry about that.)

```clojure
(def match-any-thing
  (r/rewrite
    ?x [:matched ?x])

(match-any-thing :a) ;; [:matched :a]
(match-any-thing "hello") ;; [:matched "hello"]
(match-any-thing 1) ;; [:matched 1]
```

Here we added the logic-variable `?x` to our left-hand-side. Logic variables start with an `?` and match any value. Whatever they match is now accessible on the right-hand-side. So we can match anything with `?x` and then use it in our output. Let's see a more interesting example.

```clojure
(def find-x
  (r/rewrite
   [?x] ?x
   [?x ?y] ?x
   [?x ?y ?z] ?x))

(find-x [1]) ;; 1
(find-x [1 2]) ;; 1
(find-x [1 2 3]) ;; 1
```

Here we can see some really simple rules that work on vectors of various sizes. We can use this to extract the first element from each. In this case, since we only care about `?x`, we can actually simplify this code.

```clojure
(def find-x
  (r/rewrite
   [?x] ?x
   [?x _] ?x
   [?x _ _] ?x))
```

The `_` is a wildcard match that matches anything but doesn't bind at all. What happens if we try to extend this to work for not just vectors, but just a single number?

```clojure
(def find-x
  (r/rewrite
   ?x ?x
   [?x] ?x
   [?x _] ?x
   [?x _ _] ?x))

(find-x 1) ;; 1
(find-x [1]) ;; [1]
```

The order of our rules matter, `?x` matches anything, so we will always get the first match. We could change the order, or we can constrain the match.

```clojure
(def find-x
  (r/rewrite
   (pred number? ?x) ?x
   [?x] ?x
   [?x _] ?x
   [?x _ _] ?x))

(find-x 1) ;; 1
(find-x [1]) ;; 1
```

Okay, now it works. But many of you are probably thinking "Isn't this just pattern matching?". And in many way it is. Term Rewriting is a kind of pattern matching. But it doesn't stop with simple pattern matching. Term Rewriting is a way to do all computation through pattern matching. To see that, let's move beyond the basics.

## Applying strategies

We've seen that with Meander we can do simple rewrites where we match on the left-hand-side and output a right-hand-side. But just being able to do a single rewrite in this way is really limiting. To see this problem let's consider a classic example in term rewriting.

```clojure
(def simplify-addition
  (r/rewrite
   (+ ?x 0) ?x
   (+ 0 ?x) ?x))

(simplify-addition '(+ 0 3)) ;; 3
(simplify-addition '(+ 3 0)) ;; 3
```

Zero added to anything is just that thing. We can easily express this with term rewriting. But what if we have multple 0's nested?

```clojure
(simplify-addition '(+ 0 (+ 0 3))) ;; (+ 0 3)

(simplify-addition
 (simplify-addition '(+ 0 (+ 0 3)))) ;; 3
```

As you can see, the first time we apply our ruels, we do simplify, but not all the way. If we call our rules again, we fully simplify the expression. But how could we express this with term rewriting? We can use what are called `strategies`. Strategies let use control how our terms are rewritten. Let's start with an easy strategy the `n-times` strategy.

```clojure
(def simplify-twice
  (r/n-times 2 simplify-addition))

(simplify-twice '(+ 0 (+ 0 3))) ;; 3
```

Strategies  wrap our rewriting rules and make them do additional things. In this case, the rewriting will be applied twice. But there are a few problems with the strategy as we've written it. Let's slowly discover those problems together and fix them.

```clojure
(simplify-twice '(+ 0 3)) ;; #meander.delta/fail[]
```

Our apply-twice strategy works for things that need to be simplified twice, but not for simple cases. We can fix that by using the `attempt` strategy. It will try to rewrite and if it fails, just return our value.

```clojure
(def simplify-addition
  (r/n-times 2
    (r/attempt
     (r/rewrite
      (+ ?x 0) ?x
      (+ 0 ?x) ?x))))

(simplify-addition '(+ 0 3)) ;; 3
(simplify-addition '(+ 0 (+ 0 3))) ;; 3
(simplify-addition '(+ (+ 0 (+ 0 3)) 0)) ;; (+ 0 3)
```

Now it works for both. But having it only rewrite twice is a little arbitrary. What we really want to say is to continue applying our rewrite rules until nothing changes. We can do that by using the `(until =)` strategy.

```clojure
(def simplify-addition
  (r/until =
    (r/attempt
     (r/rewrite
      (+ ?x 0) ?x
      (+ 0 ?x) ?x))))

(simplify-addition '(+ (+ 0 (+ 0 3)) 0)) ;; 3
(simplify-addition '(+ (+ 0 (+ 0 (+ 3 (+ 2 0)))) 0)) ;; (+ 3 (+ 2 0))
```

We can now simplify things no matter how deep they are, but as we can see we didn't fully eliminate 0s from all our expressions. Why is that? Well our pattern only matches things that are in the outermost expression. We don't look at all at the sub-expressions. We can fix that by applying another strategy. In this case, we will use the `bottom-up` strategy.

```clojure
(def simplify-addition
  (r/until =
    (r/bottom-up
     (r/attempt
      (r/rewrite
       (+ ?x 0) ?x
       (+ 0 ?x) ?x)))))

(simplify-addition '(+ (+ 0 (+ 0 (+ 3 (+ 2 0)))) 0)) ;; (+ 3 2)
```

We have now eliminated all the zeros in our additions no matter where they are in the tree. For the sake of space in our examples, we kept our rules and our strategies together, but these are actually separable. What if we wanted to try the `top-down` strategy with our rules?

```clojure
(def simplify-addition
  (r/rewrite
   (+ ?x 0) ?x
   (+ 0 ?x) ?x))

(def simplify-addition-bu
  (r/until =
    (r/bottom-up
     (r/attempt simplify-addition))))

(def simplify-addition-td
  (r/until =
    (r/top-down
     (r/attempt simplify-addition))))
```

Our rules are completely separate from how we want to apply them. When writing our transformations, we don't have the think at all about the context they live in. We just express our simple rules and later we can apply strategies to them. But what if we want to understand what these strategies are doing? After playing around with things, it seems that the top-down strategy and the bottom-up strategy always give us the same result. But what are they doing that is different? We can inspect our strategies at any point by using the `trace` strategy.

```clojure
(def simplify-addition-bu
  (r/until =
    (r/trace
     (r/bottom-up
      (r/attempt simplify-addition)))))

(def simplify-addition-td
  (r/until =
    (r/trace
     (r/top-down
      (r/attempt simplify-addition)))))
```

So now we have modified our rewrites to trace every time the top-down or bottom-up rules are called. Let's try a fairly complicated expression and see what happens.

```clojure
(simplify-addition-td '(+ (+ (+ 0 3) (+ 0 (+ 3 (+ 2 0)))) 0))

;; printed
{:id t_27283, :in (+ (+ (+ 0 3) (+ 0 (+ 0 (+ 2 0)))))}
{:id t_27283, :out (+ (+ 3 (+ 0 2)))}
{:id t_27283, :in (+ (+ 3 (+ 0 2)))}
{:id t_27283, :out (+ (+ 3 2))}
{:id t_27283, :in (+ (+ 3 2))}
{:id t_27283, :out (+ (+ 3 2))}



(simplify-addition-bu '(+ (+ (+ 0 3) (+ 0 (+ 3 (+ 2 0)))) 0))

;;printed
{:id t_27284, :in (+ (+ (+ 0 3) (+ 0 (+ 0 (+ 2 0)))))}
{:id t_27284, :out (+ (+ 3 2))}
{:id t_27284, :in (+ (+ 3 2))}
{:id t_27284, :out (+ (+ 3 2))}
```

If we look at the top-down approach, we can see that the top-down strategy actually gets called three times. Once it rewrites quite a bit, but leaves in a 0 that needs to be rewritten. Then it gets called again, eliminating all zeros. Finally it is called and nothing changes. Our bottom-up strategy however is only called twice. But we can actually get more fine grained than this. We can put trace at any point in our strategies.

```clojure
(def simplify-addition-bu
  (r/until =
    (r/bottom-up
     (r/trace
      (r/attempt simplify-addition)))))

(simplify-addition-bu '(+ (+ 0 3) 0))

;; printed
{:id t_27317, :in +}
{:id t_27317, :out +}
{:id t_27317, :in +}
{:id t_27317, :out +}
{:id t_27317, :in 0}
{:id t_27317, :out 0}
{:id t_27317, :in 3}
{:id t_27317, :out 3}
{:id t_27317, :in (+ 0 3)}
{:id t_27317, :out 3}
{:id t_27317, :in 0}
{:id t_27317, :out 0}
{:id t_27317, :in (+ 3 0)}
{:id t_27317, :out 3}
{:id t_27317, :in 3}
{:id t_27317, :out 3}
```

Here we moved our trace down outside our `attempt` strategy. Now we can see the exact order of our bottom-up strategy. Having this sorts of visibility into how the process is working is really fantastic.

## Rewriting as General Computation

What have been doing so far is interesting, but it falls short of the true power of term rewriting. Term rewriting is a general programming technique. Using it we can compute absolutely anything that is computable. Let's start with a classic example, fibonacci, but to prove general computability, we will make our own numbers instead relying on Clojure's.

```clojure
(def fib-rules
  (r/rewrite

   (+ Z ?n) ?n
   (+ ?n Z) ?n

   (+ ?n (succ ?m)) (+ (succ ?n) ?m)
   
   (fib Z) Z
   (fib (succ Z)) (succ Z)
   (fib (succ (succ ?n))) (+ (fib (succ ?n)) (fib ?n))))


(def run-fib
  (r/until =
    (r/bottom-up
     (r/attempt fib-rules))))

[(run-fib '(fib Z))
 (run-fib '(fib (succ Z)))
 (run-fib '(fib (succ (succ Z))))
 (run-fib '(fib (succ (succ (succ Z)))))
 (run-fib '(fib (succ (succ (succ (succ Z))))))
 (run-fib '(fib (succ (succ (succ (succ (succ Z)))))))
 (run-fib '(fib (succ (succ (succ (succ (succ (succ Z))))))))]

;; [Z
;;  (succ Z)
;;  (succ Z)
;;  (succ (succ Z))
;;  (succ (succ (succ Z)))
;;  (succ (succ (succ (succ (succ Z)))))
;;  (succ (succ (succ (succ (succ (succ (succ (succ Z))))))))]
```

If you aren't familiar with defining natural numbers via Peano numbers this may be a little bit confusion. But for our purposes all you need to know is that `Z` means 0 and `succ` means successor. `(succ Z)` means 1 `(succ (succ Z))` means 2 and so on and so forth. Our fibonacci rules start by defining addition for our peano numbers. Anything added to 0 is zero. Otherwise, we can add two numbers my moving all the `succ`s to one side until the right hand side equals 0. With those definitions in place, 

