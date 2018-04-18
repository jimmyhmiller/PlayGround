try {
	const { green, red, bold, underline } = require('colors/safe');
	const assert = require('assert');
	const highlight = require("@babel/highlight").default;
	const intercept = require("intercept-stdout");
		const fs = require('fs');

	 

	const frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
	let i = 0;

	let captureOutput = true;

	const output = [];

	const outputCapturer = (f) => (txt) => {
		if (captureOutput) {
			output.push(f(txt))
			return undefined;
		}
		return f(txt)
	}

	const unhook_intercept = intercept(outputCapturer(function(txt) {
		if (!txt) {
			return;
		} 
		if (i === 4) {
			return txt
		}
		return txt.replace("✔" , frames[i])
			      .replace("❌", frames[i])
			      .replace(/➡.*/, frames[i]);
	}));

	

	const { head, emptyList, prepend, tail, isEmpty, map, filter, list, not, myExample, myFunc, zero, one, inc, dec, toNum, add, sub, fromNum, mult, nth, countDown, length, isTrue, last, showAll } = require('./list7.js')


	
	captureOutput = false;

	const printDivider = () => {
		console.log("");
		console.log("━".repeat(process.stdout.columns))
		console.log("");
	}

	const runTest = (test) => {
		try {
			const value = eval(`assert(${test})`);
			console.log(`${highlight(test)} ${green("✔")}`)
		}
		catch (e) {

			if (!showAll && e && e.message && e.message.endsWith("is not a function")) {
				return;
			}
			console.log(`${highlight(test)} ${red("❌")}`)
			if ( !(e instanceof assert.AssertionError)) {
				console.log(e.message)
			}
		}
	}

	const runExample = (example) => {
		try {
			const value = eval(example);
			console.log(`${highlight(example)} ➡ ${value}`)
		}
		catch (e) {
			console.log(`${highlight(example)} ➡ ${e.message}`)
		}
	}

	const myTests = [
	  "myFunc() === 42",
	  "head(prepend(1, emptyList())) === 1",
	  "head(prepend(2, emptyList())) === 2",
	  "head(tail(prepend(1, prepend(2, emptyList())))) === 2",
	  "isTrue(isEmpty(emptyList()))",
	  "isTrue(not(isEmpty(prepend(1, emptyList()))))",
  	  "last(prepend(1, prepend(2, emptyList()))) === 2",
	  "toNum(zero) === 0",
	  "toNum(one) === 1",
	  "toNum(inc(one)) === 2",
	  "toNum(dec(one)) === 0",
	  "toNum(fromNum(4)) === 4",
	  "toNum(add(fromNum(2), fromNum(2))) === 4",
	  "toNum(sub(fromNum(2), fromNum(2))) === 0",
	  "toNum(mult(fromNum(3), fromNum(2))) === 6",
	  "toNum(last(countDown(fromNum(5)))) === 0",
	  "toNum(nth(countDown(fromNum(5)), fromNum(5))) === 0",
  	  "toNum(length(countDown(fromNum(100)))) === 101",
	]

	Array.from({length: 5}, (x,i) => i).forEach(j => {
		setTimeout(function() {
			i = j;
			console.log('\033c')
			console.log(bold(underline("TESTS")))
			console.log("")
			myTests.forEach(runTest)

			const contents = fs.readFileSync('list7.js', 'utf8');
			printDivider()
			console.log(bold(underline("OUTPUT")))
			console.log("")
			output.forEach(s => console.log(s))
			const notes = contents.split("\n")
				   .filter(s => s.startsWith("//"))
				   .map(s => s.substring(3));
			printDivider()
			console.log(bold(underline("NOTES")))
			console.log("");
			// if (notes.length > 0 && notes[0].trim().toLowerCase().startsWith("notes")) {
			notes.forEach(x => console.log(x))
			console.log("");
			// }


		}, 80*j);

	})
} catch(e) {
	console.log('\033c');
	console.error(e);
}



