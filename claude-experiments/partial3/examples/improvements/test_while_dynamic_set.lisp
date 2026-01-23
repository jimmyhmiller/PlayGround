; Regression test - while loop with dynamic set! should preserve assignments
(let arr y
  (let result 0
    (let state 100
      (begin
        (while (> state 0)
          (begin
            (set! result (index arr 0))
            (set! state -1)))
        result))))
