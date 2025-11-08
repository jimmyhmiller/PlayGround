# Test for expressions with function call as iterator type
# From src/arr/trove/checker.arr line 826

fun make-iter(loc):
  map
end

result = for make-iter(loc)(lv from [list: 1, 2, 3], rv from [list: 4, 5, 6]):
  lv + rv
end
