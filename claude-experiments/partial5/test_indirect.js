// Test case that triggers path splitting with side effects in condition
// Pattern: ternary assignment where condition has side effects (--sp)
// and the condition result depends on dynamic data

function makeHandler() {
    var stack = [0, 0, 0, 0];
    var sp = 1;  // Points to index 1
    var state;

    // Put a dynamic value at stack[0]
    stack[0] = Date.now() > 1000000000000 ? 1 : 0;  // Dynamic 0 or 1

    // Critical pattern: --sp makes sp=0, then reads stack[0] which is dynamic
    state = stack[--sp] ? 200 : 100;

    // Simple state machine
    while (state >= 0) {
        switch (state) {
            case 100:
                console.log("took false branch");
                state = -1;
                break;
            case 200:
                console.log("took true branch");
                state = -1;
                break;
        }
    }
}

document.addEventListener("click", makeHandler);
