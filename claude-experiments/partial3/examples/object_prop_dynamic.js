// Test: object with dynamic property, then access a static property
var arr = [1, 2, 3];
var obj = {
    _data: arr,
    _dynamic: dynamicVal
};

// Access the static property - should resolve
var data = obj._data;
var len = data.length;

// Access the dynamic property - should remain in residual
var dyn = obj._dynamic;

len + dyn;
