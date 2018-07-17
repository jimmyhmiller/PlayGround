(ns test-unify
  (:require [unify :refer :all]))




(comment

  (-> {}
      (unify 'x 2)
      (unify 'y 'x)
      (unify 'z 'y)
      (lookup 'z))

  (-> {}
      (unify 'x 'y)
      (unify 'y 'z)
      (unify 'z 2)
      (lookup 'x))


  (-> {}
      (unify 2 'x)
      (unify 'x 'y)
      (unify 'z 'y)
      (lookup 'z)))















;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;









(comment
  (match true
         true false
         false true)

  (match :name
         :otherName :otherThing
         :name :thing)


  (match [1 2 3]
         [x]     {:x x}
         [x y]   {:x x :y y}
         [x y z] {:x x :y y :z z})


  (match [1 2 1]
         [x y x] {:x x :y y}
         [x y z] {:x x :y y :z z})


  (defmatch fib
    [0] 0
    [1] 1
    [n] (+ (fib (- n 1))
           (fib (- n 2))))



  (defmatch get-x
    [x] x
    [x y] x
    [x y z] x)

  (get-x 1)
  (get-x 1 2)
  (get-x 1 2 3))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;












(comment

  (def db
    [[1 :age 26]
     [1 :name "jimmy"]
     [2 :age 26]
     [2 :name "steve"]
     [3 :age 24]
     [3 :name "bob"]
     [4 :address 1]
     [4 :address-line-1 "123 street st"]
     [4 :city "Indianapolis"]])



  (q {:find {:name name}
      :where [[_ :name name]]}
     db)


  (q {:find {:name name
             :age age}
      :where [[e :name name]
              [e :age age]]}
     db)




  (q {:find {:name1 name1
             :name2 name2}
      :where [[e1 :name name1]
              [e2 :name name2]
              [e1 :age age]
              [e2 :age age]]}
     db)



  (q {:find {:name name
             :address-line-1 address-line-1
             :city city}
      :where [[e :name name]
              [a :address e]
              [a :address-line-1 address-line-1]
              [a :city city]]}
     db))
