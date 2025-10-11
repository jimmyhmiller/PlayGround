(ns test)

(def Color (: Type) (Enum Red Green Blue))

(def get_color_value (: (-> [Color] Int))
  (fn [c]
    (if (= c Color/Red)
      1
      (if (= c Color/Green)
        2
        3))))

(def result1 (: Int) (get_color_value Color/Red))
(def result2 (: Int) (get_color_value Color/Green))
(def result3 (: Int) (get_color_value Color/Blue))
(printf (c-str "%lld %lld %lld\n") result1 result2 result3)
