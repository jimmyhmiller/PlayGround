function makeCounter() {
	var value = 0;
	return function (property) {
		if (property === 'increment') {
			return function () {
				value += 1
				return value
			}
		}
	}
}

var counter1 = makeCounter();
var counter2 = makeCounter();


counter1('increment')() // 1
counter1('increment')() // 2

counter2('increment')() // 1
counter1('increment')() // 3
