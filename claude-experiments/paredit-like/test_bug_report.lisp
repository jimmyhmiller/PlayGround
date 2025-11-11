                (region
                  (block
                    (arguments [])
                    (declare c1_rec (arith.constant {:value (: 1 i32)}))
                    (declare n_minus_1 (arith.subi %n %c1_rec))
                    (operation
                      (name scf.yield)
                      (operands %n_minus_1)))))))

            (func.return %result))))))
