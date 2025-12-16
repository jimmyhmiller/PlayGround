;;; hot-reload.el --- Expression-level hot reloading for JavaScript -*- lexical-binding: t; -*-

;; Copyright (C) 2024

;; Author: Hot Reload
;; Version: 0.1.0
;; Package-Requires: ((emacs "27.1") (websocket "1.13"))
;; Keywords: javascript, hot-reload, repl
;; URL: https://github.com/example/hot-reload

;;; Commentary:

;; This package provides expression-level hot reloading for JavaScript,
;; similar to CIDER/Calva for Clojure. Evaluate functions, variables,
;; or expressions directly into a running Node.js process.
;;
;; Usage:
;;   1. Start your app with `hot run ./app.js`
;;   2. M-x hot-reload-connect
;;   3. Place cursor on a function and M-x hot-reload-eval-defun
;;
;; Keybindings (suggested):
;;   C-c C-c   hot-reload-eval-defun
;;   C-x C-e   hot-reload-eval-last-sexp
;;   C-c C-r   hot-reload-eval-region

;;; Code:

(require 'websocket)
(require 'json)
(require 'js)

(defgroup hot-reload nil
  "Expression-level hot reloading for JavaScript."
  :prefix "hot-reload-"
  :group 'tools)

(defcustom hot-reload-port 3456
  "Port for the hot reload WebSocket server."
  :type 'integer
  :group 'hot-reload)

(defcustom hot-reload-host "localhost"
  "Host for the hot reload WebSocket server."
  :type 'string
  :group 'hot-reload)

(defcustom hot-reload-show-result-in-buffer nil
  "If non-nil, show results in a dedicated buffer instead of minibuffer."
  :type 'boolean
  :group 'hot-reload)

(defvar hot-reload--websocket nil
  "The WebSocket connection to the hot reload server.")

(defvar hot-reload--pending-evals (make-hash-table :test 'equal)
  "Hash table of pending eval requests by request ID.")

(defvar hot-reload--request-counter 0
  "Counter for generating unique request IDs.")

(defvar hot-reload--source-root nil
  "The source root directory for computing module IDs.")

(defun hot-reload--generate-request-id ()
  "Generate a unique request ID."
  (format "emacs-%d-%d" (emacs-pid) (cl-incf hot-reload--request-counter)))

(defun hot-reload--get-module-id ()
  "Get the module ID for the current buffer.
This is the relative path from the source root."
  (if (and buffer-file-name hot-reload--source-root)
      (file-relative-name buffer-file-name hot-reload--source-root)
    (and buffer-file-name (file-name-nondirectory buffer-file-name))))

(defun hot-reload--on-message (_websocket frame)
  "Handle incoming WebSocket FRAME."
  (let* ((text (websocket-frame-text frame))
         (json-object-type 'plist)
         (json-array-type 'list)
         (msg (json-read-from-string text))
         (type (plist-get msg :type)))
    (cond
     ((string= type "eval-result")
      (hot-reload--handle-eval-result msg)))))

(defun hot-reload--handle-eval-result (msg)
  "Handle an eval result message MSG."
  (let* ((request-id (plist-get msg :requestId))
         (success (plist-get msg :success))
         (value (plist-get msg :value))
         (error-msg (plist-get msg :error))
         (module-id (plist-get msg :moduleId))
         (expr-type (plist-get msg :exprType))
         (callback (gethash request-id hot-reload--pending-evals)))
    ;; Remove from pending
    (remhash request-id hot-reload--pending-evals)
    ;; Display result
    (if success
        (let ((result-str (hot-reload--format-value value)))
          (if hot-reload-show-result-in-buffer
              (hot-reload--show-in-buffer result-str module-id)
            (message "[hot] %s => %s" module-id result-str))
          (when callback (funcall callback value)))
      (message "[hot] Error in %s: %s" module-id error-msg))))

(defun hot-reload--format-value (value)
  "Format VALUE for display."
  (cond
   ((null value) "null")
   ((eq value t) "true")
   ((eq value :json-false) "false")
   ((stringp value) value)
   ((numberp value) (number-to-string value))
   (t (json-encode value))))

(defun hot-reload--show-in-buffer (result module-id)
  "Show RESULT in a dedicated buffer for MODULE-ID."
  (let ((buf (get-buffer-create "*hot-reload*")))
    (with-current-buffer buf
      (goto-char (point-max))
      (insert (format "\n;; %s @ %s\n%s\n"
                      module-id
                      (format-time-string "%H:%M:%S")
                      result)))
    (display-buffer buf)))

(defun hot-reload--on-close (_websocket)
  "Handle WebSocket close."
  (setq hot-reload--websocket nil)
  (message "[hot] Disconnected from server"))

(defun hot-reload--on-error (_websocket _type err)
  "Handle WebSocket error ERR."
  (message "[hot] WebSocket error: %s" err))

;;;###autoload
(defun hot-reload-connect (&optional source-root)
  "Connect to the hot reload server.
SOURCE-ROOT is the root directory for computing module IDs.
If not provided, uses the project root or current directory."
  (interactive
   (list (read-directory-name "Source root: "
                              (or (locate-dominating-file default-directory "package.json")
                                  default-directory))))
  (when hot-reload--websocket
    (websocket-close hot-reload--websocket))
  (setq hot-reload--source-root (expand-file-name source-root))
  (let ((url (format "ws://%s:%d" hot-reload-host hot-reload-port)))
    (message "[hot] Connecting to %s..." url)
    (setq hot-reload--websocket
          (websocket-open
           url
           :on-message #'hot-reload--on-message
           :on-close #'hot-reload--on-close
           :on-error #'hot-reload--on-error
           :on-open (lambda (_ws)
                      (message "[hot] Connected to %s" url)
                      ;; Identify as editor
                      (hot-reload--send
                       '(:type "identify" :clientType "editor")))))))

(defun hot-reload-disconnect ()
  "Disconnect from the hot reload server."
  (interactive)
  (when hot-reload--websocket
    (websocket-close hot-reload--websocket)
    (setq hot-reload--websocket nil)
    (message "[hot] Disconnected")))

(defun hot-reload--send (msg)
  "Send MSG to the hot reload server."
  (when hot-reload--websocket
    (websocket-send-text hot-reload--websocket (json-encode msg))))

(defun hot-reload--eval (expr &optional callback)
  "Evaluate EXPR in the current module's context.
CALLBACK is called with the result if provided."
  (unless hot-reload--websocket
    (error "[hot] Not connected. Run M-x hot-reload-connect first"))
  (let* ((module-id (hot-reload--get-module-id))
         (request-id (hot-reload--generate-request-id)))
    (when callback
      (puthash request-id callback hot-reload--pending-evals))
    (hot-reload--send
     `(:type "eval-request"
       :moduleId ,module-id
       :expr ,expr
       :requestId ,request-id))
    (message "[hot] Evaluating in %s..." module-id)))

;;;###autoload
(defun hot-reload-eval-defun ()
  "Evaluate the top-level form at point.
For JavaScript, this evaluates the function, class, or variable
declaration containing point."
  (interactive)
  (save-excursion
    (let ((start (progn (js-beginning-of-defun) (point)))
          (end (progn (js-end-of-defun) (point))))
      (hot-reload--eval (buffer-substring-no-properties start end)))))

;;;###autoload
(defun hot-reload-eval-last-sexp ()
  "Evaluate the expression before point.
Tries to find a complete expression (statement or expression)."
  (interactive)
  (let ((end (point))
        start)
    (save-excursion
      ;; Try to find statement start
      (backward-sexp)
      (setq start (point)))
    (hot-reload--eval (buffer-substring-no-properties start end))))

;;;###autoload
(defun hot-reload-eval-region (start end)
  "Evaluate the region from START to END."
  (interactive "r")
  (hot-reload--eval (buffer-substring-no-properties start end)))

;;;###autoload
(defun hot-reload-eval-buffer ()
  "Evaluate the entire buffer as a module reload."
  (interactive)
  (hot-reload--eval (buffer-substring-no-properties (point-min) (point-max))))

;;;###autoload
(defun hot-reload-eval-expression (expr)
  "Evaluate EXPR in the current module's context."
  (interactive "sEval: ")
  (hot-reload--eval expr))

;; Minor mode for convenient keybindings
(defvar hot-reload-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "C-c C-c") #'hot-reload-eval-defun)
    (define-key map (kbd "C-x C-e") #'hot-reload-eval-last-sexp)
    (define-key map (kbd "C-c C-r") #'hot-reload-eval-region)
    (define-key map (kbd "C-c C-k") #'hot-reload-eval-buffer)
    (define-key map (kbd "C-c C-q") #'hot-reload-disconnect)
    map)
  "Keymap for `hot-reload-mode'.")

;;;###autoload
(define-minor-mode hot-reload-mode
  "Minor mode for expression-level JavaScript hot reloading.

\\{hot-reload-mode-map}"
  :lighter " Hot"
  :keymap hot-reload-mode-map
  (if hot-reload-mode
      (message "[hot] Hot reload mode enabled. C-c C-c to eval defun.")
    (message "[hot] Hot reload mode disabled.")))

(provide 'hot-reload)

;;; hot-reload.el ends here
