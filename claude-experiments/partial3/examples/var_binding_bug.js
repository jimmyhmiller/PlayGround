// Test with while loop and switch

(function() {
  var v7 = 1;
  var v18;

  while (v7 >= 0) {
    switch (v7 & 1) {
      case 1:
        v18 = [];
        v18[1] = 21;
        v18[2] = 35;
        v7 = -1;
        break;
    }
  }

  return v18[1];
})();
