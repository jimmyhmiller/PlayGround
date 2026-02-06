// Simple VM with while-switch pattern
// This tests the partial evaluator's ability to collapse control flow
// when the state is known but input is dynamic.

function runVM(program, input) {
    var state = 0;
    var stack = [];
    var result = null;

    while (state >= 0) {
        switch (state) {
            case 0:
                // PUSH input onto stack
                stack.push(input);
                state = 1;
                break;
            case 1:
                // LOG - side effect!
                console.log("Processing:", stack[0]);
                state = 2;
                break;
            case 2:
                // COMPUTE - multiply by 2
                result = stack[0] * 2;
                state = 3;
                break;
            case 3:
                // OUTPUT - side effect!
                console.log("Result:", result);
                state = -1;
                break;
        }
    }
    return result;
}

// Event handler that uses the VM
// 'program' is known (empty array), 'input' is dynamic (event.clientX)
document.addEventListener("click", function(event) {
    runVM([], event.clientX);
});
