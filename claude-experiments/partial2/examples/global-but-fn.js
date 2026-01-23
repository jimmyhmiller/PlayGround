var x = [[1]];
function returnX() {
	return x[0][0];
}

var a = returnX()
x[0][0] = 42;
var b = returnX();

console.log(a, b)