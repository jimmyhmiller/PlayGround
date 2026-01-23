// Test case: return should emit the variable name, not the assigned expression
// When v44 is set to a dynamic expression, `return v44` should emit `v44`, not the whole expression

var v9 = dynamicArray;  // dynamic
var v10 = 0;
var v11 = 0;

function v4() {
    var v43 = 200;
    var v44;
    var v45;
    while (v43 >= 0) {
        switch (v43 & 1) {
            case 0:
                v45 = v9[v10];
                v11++;
                v44 = (v45[v11++] << 24) | (v45[v11++] << 16) | (v45[v11++] << 8) | v45[v11++];
                v43 = -1;
                break;
            case 1:
                v43 = -1;
                break;
        }
    }
    return v44;
}

v4();
