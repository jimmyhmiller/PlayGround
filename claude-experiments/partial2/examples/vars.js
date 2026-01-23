(function() {


	var x1, x2, x3;

	function myFunction() {
		x1 = [[42]];
		x2 = 0;
		x3 = x1[x2][x2];
	}

	myFunction();

	console.log(x1);
	console.log(x2);
	console.log(x3);
})()



var x = function() {


	var x1, x2, x3;

	function myFunction() {
		x1 = [[42]];
		x2 = 0;
		x3 = x1[x2][x2];
	}

	myFunction();

	console.log(x1);
	console.log(x2);
	console.log(x3);
};

x();