;;; coil-mode.el --- Major mode + inferior REPL for the Coil language -*- lexical-binding: t; -*-

;; A minimal major mode for .coil files (a `lisp-mode' derivative), plus a
;; `run-coil' command that starts `coil repl' through Emacs' stock
;; inferior-lisp machinery — so the classic Lisp keys work unchanged:
;;
;;   C-x C-e   lisp-eval-last-sexp    send the sexp before point
;;   C-M-x     lisp-eval-defun        send the whole top-level form
;;   C-c C-r   lisp-eval-region       send the region
;;   C-c C-l   lisp-load-file         :load a whole file into the session
;;   C-c C-z   switch-to-lisp         jump to the REPL buffer
;;
;; Setup (in your init file):
;;
;;   (add-to-list 'load-path "/path/to/coil/emacs")
;;   (require 'coil-mode)
;;   ;; if `coil' isn't on PATH:
;;   ;; (setq coil-program "/path/to/coil/target/release/coil")
;;
;; Then open a .coil file and M-x run-coil.
;;
;; REPL semantics worth knowing (see `coil repl`'s :help): top-level
;; definitions persist in the session and REDEFINE by name — C-M-x on an
;; edited defn replaces it. Any other form is an expression: compiled with the
;; session's definitions, run, and its value printed by inferred type. The
;; session is LIVE by default (each eval is a dylib hot-loaded into the
;; session process), so `(def x EXPR)' binds values that later sends can use,
;; and malloc'd/FFI state survives across sends. A crashing eval kills the
;; REPL process (restart with M-x run-coil); `coil repl --isolate' trades the
;; persistence for per-expression crash isolation.

;;; Code:

(require 'inf-lisp)

(defgroup coil nil
  "The Coil language."
  :group 'languages)

(defcustom coil-program "coil"
  "The coil compiler executable (`coil repl' must work with it)."
  :type 'string
  :group 'coil)

(defconst coil--definition-forms
  '("defn" "defstruct" "defsum" "deftrait" "impl" "extern" "const" "defcc"
    "import" "include" "export" "export-c" "static-assert" "meta" "derive"
    "module")
  "Coil's top-level definition heads.")

(defconst coil--core-forms
  '("if" "do" "let" "match" "loop" "break" "continue" "while" "for" "for-in"
    "when" "cond" "case" "block" "return-from" "defer" "quote" "quasiquote"
    "cast" "field" "index" "load" "store!" "alloc-stack" "alloc-heap"
    "alloc-static" "sizeof" "fnptr-of" "llvm-ir" "comptime")
  "Core / everyday special forms, for highlighting.")

(defconst coil-font-lock-keywords
  `((,(concat "(" (regexp-opt coil--definition-forms t) "\\_>")
     (1 font-lock-keyword-face))
    ;; the name being defined: (defn NAME …), (defstruct NAME …), …
    (,(concat "(" (regexp-opt '("defn" "defcc") t) "\\_>[ \t]+\\(\\_<[^ \t\n)]+\\)")
     (2 font-lock-function-name-face))
    (,(concat "(" (regexp-opt '("defstruct" "defsum" "deftrait") t)
              "\\_>[ \t]+\\(\\_<[^ \t\n)]+\\)")
     (2 font-lock-type-face))
    (,(concat "(" (regexp-opt coil--core-forms t) "\\_>")
     (1 font-lock-builtin-face))
    ;; :keywords (types like :i64, spec args, import flags)
    ("\\_<:[^ \t\n()\"]+" . font-lock-constant-face))
  "Font-lock for `coil-mode', layered over `lisp-mode's.")

;;;###autoload
(define-derived-mode coil-mode lisp-mode "Coil"
  "Major mode for editing Coil source files.

\\{coil-mode-map}"
  (font-lock-add-keywords nil coil-font-lock-keywords)
  (setq-local inferior-lisp-program (concat coil-program " repl"))
  ;; C-c C-l sends `:load FILE' (the REPL's whole-file command) instead of
  ;; the default `(load "FILE")'.
  (setq-local inferior-lisp-load-command ":load %s\n")
  (setq-local comment-start "; ")
  (setq-local comment-add 1))

;; Definition forms indent like defuns.
(dolist (head coil--definition-forms)
  (put (intern head) 'lisp-indent-function 'defun))
(dolist (head '("let" "loop" "while" "for" "for-in" "when" "match" "do" "block"))
  (put (intern head) 'lisp-indent-function 1))

;;;###autoload
(defun run-coil ()
  "Run an inferior Coil REPL (`coil repl'), or switch to the running one.
Everything `inferior-lisp' offers applies: the buffer is *inferior-lisp*,
and the lisp-mode eval keys in .coil buffers send to it."
  (interactive)
  (inferior-lisp (concat coil-program " repl")))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.coil\\'" . coil-mode))

(provide 'coil-mode)
;;; coil-mode.el ends here
