; Test case for Improvement #5: Opaque Constructor Handling
; Built-in constructors (Uint8Array, ArrayBuffer, DataView, TextDecoder)
; should be handled specially to enable further optimization.
;
; Run with: cargo run -- examples/improvements/test_5_opaque_builtins.lisp --builtins

; Test 1: ArrayBuffer with known size should have byteLength accessible
(let buf (new ArrayBuffer 16)
  (prop buf "byteLength"))

; Expected: 16 (fully static)

; Test 2: DataView on ArrayBuffer should allow reading static values
(let buf (new ArrayBuffer 8)
  (let dv (new DataView buf)
    (begin
      ; Set some bytes
      (call (prop dv "setUint8") 0 65)
      (call (prop dv "setUint8") 1 66)
      ; Read them back
      (call (prop dv "getUint8") 0))))

; Expected: 65 (fully static)

; Test 3: TextDecoder with known bytes should decode statically
(let buf (new ArrayBuffer 5)
  (let dv (new DataView buf)
    (begin
      (call (prop dv "setUint8") 0 72)  ; H
      (call (prop dv "setUint8") 1 101) ; e
      (call (prop dv "setUint8") 2 108) ; l
      (call (prop dv "setUint8") 3 108) ; l
      (call (prop dv "setUint8") 4 111) ; o
      (let ua (new Uint8Array buf)
        (let td (new TextDecoder "utf-8")
          (call (prop td "decode") ua))))))

; Expected: "Hello" (fully static)
