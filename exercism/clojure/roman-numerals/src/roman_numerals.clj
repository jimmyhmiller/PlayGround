(ns roman-numerals)

(def numeral-info [["M" 1000] ["CM" 900] ["D" 500] ["CD" 400] ["C" 100] 
               ["XC" 90] ["L" 50] ["XL" 40] ["X" 10] ["IX" 9] 
               ["V" 5] ["IV" 4] ["I" 1]])

(defn apply-numeral [[current-roman remaining-decimal] [numeral value]]
  (if (< remaining-decimal value)
    [current-roman remaining-decimal]
    (apply-numeral [(str current-roman numeral) (- remaining-decimal value)] 
                   [numeral value])))

(defn numerals [num]
  (first (reduce apply-numeral ["" num] numeral-info)))

