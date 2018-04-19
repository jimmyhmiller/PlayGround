(ns enso.AXValueRef)

(import '[com.sun.jna Library Native NativeLibrary PointerType Pointer])

(gen-class
 :name "enso.AXValueRef"
 :extends enso.CFTypeRef)
