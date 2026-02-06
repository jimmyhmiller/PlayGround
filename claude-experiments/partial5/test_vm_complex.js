// More complex VM test with multiple side effects
// Tests: array mutations, object assignments, dynamic callbacks

var results = [];
var config = {
    multiplier: 3,
    offset: 10
};

function runComplexVM(opcodes, input) {
    var state = 0;
    var pc = 0;
    var acc = 0;
    var temp = null;

    while (state >= 0) {
        switch (state) {
            case 0:
                // LOAD - load input into accumulator
                acc = input;
                state = 1;
                break;
            case 1:
                // LOG input
                console.log("Input loaded:", acc);
                state = 2;
                break;
            case 2:
                // MULTIPLY by config.multiplier
                acc = acc * config.multiplier;
                state = 3;
                break;
            case 3:
                // LOG after multiply
                console.log("After multiply:", acc);
                state = 4;
                break;
            case 4:
                // ADD offset
                acc = acc + config.offset;
                state = 5;
                break;
            case 5:
                // STORE result
                results.push(acc);
                state = 6;
                break;
            case 6:
                // LOG final
                console.log("Final result:", acc);
                state = -1;
                break;
        }
    }
    return acc;
}

// Test with mouse events
document.addEventListener("mousemove", function(event) {
    runComplexVM([], event.pageX);
});

// Test with keyboard events
document.addEventListener("keydown", function(event) {
    runComplexVM([], event.keyCode);
});
