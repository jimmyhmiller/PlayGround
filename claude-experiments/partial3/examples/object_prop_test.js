// Test: object property access should be optimized when the object is known
var arr = [1, 2, 3];
var obj = {
    _push: function(x) { arr.push(x); },
    _data: arr
};

// This should optimize to access the known array
var data = obj._data;
var len = data.length;

len;
