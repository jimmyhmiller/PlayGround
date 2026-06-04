function main(input) {
  var a = [10, 20, 30, 40];
  var read = function () { return a[input]; };   // captures a -> a is boxed
  var n = input;
  var sum = 0;
  while (n > 0) {
    n = n - 1;
    a[input] = a[input] + 1;     // write in main
    sum = sum + read();          // read via closure
  }
  return sum;
}
