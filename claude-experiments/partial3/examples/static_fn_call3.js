// Test case: function with captured mutations should still be inlineable when called
// This simulates the VM pattern where a helper function reads bytes and updates state

var v0 = (function() {
    var data = [1, 2, 3, 4];  // static array
    var index = 0;
    var checksum = 0;

    function readByte() {
        var byte = data[index];
        index = index + 1;
        checksum = (checksum + byte) & 255;
        return byte;
    }

    // These should be inlined even though readByte mutates captured vars
    var state = 100;
    var result;
    while (state >= 0) {
        switch (state & 1) {
            case 0:
                result = readByte();  // Should be inlined!
                state = -1;
                break;
            case 1:
                state = -1;
                break;
        }
    }
    return result;
})();
