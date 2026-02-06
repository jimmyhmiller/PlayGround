// Pattern matching simple_unique.js: v51 = v5(); while(v51) v51 = v51();
// The branching happens INSIDE v5(), not in the assignment

function makeHandler(event) {
    var data = event.x;

    function pathA() {
        console.log("A:", data);
        return null;
    }

    function pathB() {
        console.log("B:", data * 2);
        return null;
    }

    // v5-like function that returns a continuation based on dynamic condition
    function getStartContinuation() {
        if (Date.now() > 0) {
            return pathA;
        } else {
            return pathB;
        }
    }

    // The simple_unique.js pattern
    var cont = getStartContinuation();
    while (cont) {
        cont = cont();
    }
}

document.addEventListener("click", makeHandler);
