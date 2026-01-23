// Simpler test: switch with static discriminant but dynamic set! in body
var arr = dynamicArray;
var x = 0;
var result;

switch (x) {
    case 0:
        result = arr[0];
        x = -1;
        break;
}

result;
