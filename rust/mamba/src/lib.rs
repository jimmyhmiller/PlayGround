mod bindings;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use crate::bindings::{PyFunctionObject, PyObject_Repr, PyBytes_AsString, PyUnicode_AsEncodedString};
#[no_mangle]
extern "C" fn jit_compile(func: PyFunctionObject) {
    unsafe {

        // PyObject* repr = PyObject_Repr(obj);
        // PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
        // const char *bytes = PyBytes_AS_STRING(str);
        let repr = PyObject_Repr(func.func_name);
        let utf8 = CString::new("utf-8").unwrap();
        let e = CString::new("~E~").unwrap();
        let py_str = PyUnicode_AsEncodedString(repr, utf8.as_ptr(), e.as_ptr());
        let py_str = PyBytes_AsString(py_str);
        if py_str.is_null() {
            return
        }
        let c_str = CStr::from_ptr(py_str);
        let string = c_str.to_str().unwrap().to_owned();

        println!("Compiling! {:?}", string);
    }
}


