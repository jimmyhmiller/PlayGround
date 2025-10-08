;; ============================================================================
;; MODULAR BOOTSTRAP - Import-Based System
;; ============================================================================
;; This bootstrap file demonstrates the import system.
;; Instead of defining everything in one file, we import from separate modules!

;; Import the core Lisp dialect definition
(import lisp-core)

;; Import optimization patterns
(import optimizations)

;; Import lowering transformations
(import lowering)

;; ============================================================================
;; Bootstrap Complete!
;; ============================================================================
;; The compiler is now fully loaded from separate modules.
;; Each module is self-contained and can be reused.
