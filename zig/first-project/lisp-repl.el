;;; lisp-repl.el --- Emacs configuration for our custom Lisp REPL with inline evaluation

;; Usage:
;; 1. Load this file: M-x load-file RET lisp-repl.el RET
;; 2. Start the REPL: M-x run-lisp RET
;; 3. Open a .lisp file and use:
;;    - C-x C-e : evaluate expression before point (inline result)
;;    - C-M-x   : evaluate current defun (inline result)
;;    - C-c C-r : evaluate region
;;    - C-c C-l : load file
;;    - C-c C-z : switch to REPL buffer

(require 'lisp-mode)
(require 'inf-lisp)

;; Set the path to your REPL executable
(setq inferior-lisp-program
      (expand-file-name "repl"
                        (file-name-directory load-file-name)))

;; Set the prompt regexp to match "> " and "  " (continuation prompt)
(setq inferior-lisp-prompt "^\\(?:>\\|  \\) *")

;; Optional: define a major mode for .lisp files that uses this REPL
(add-to-list 'auto-mode-alist '("\\.lisp\\'" . lisp-mode))

;;; Inline evaluation (CIDER-style)

(defvar lisp-repl-overlay nil
  "Overlay for displaying inline evaluation results.")

(defface lisp-repl-result-face
  '((((class color) (background light))
     :foreground "ForestGreen" :box (:line-width -1 :color "lightgreen"))
    (((class color) (background dark))
     :foreground "#859900" :box (:line-width -1 :color "#2aa198")))
  "Face for displaying evaluation results.")

(defun lisp-repl--remove-result-overlay ()
  "Remove the inline result overlay."
  (when (overlayp lisp-repl-overlay)
    (delete-overlay lisp-repl-overlay))
  (setq lisp-repl-overlay nil)
  (remove-hook 'pre-command-hook #'lisp-repl--remove-result-overlay))

(defun lisp-repl--make-result-overlay (value where duration)
  "Create an overlay showing VALUE at WHERE with DURATION.
WHERE should be a position. DURATION can be 'command or a number of seconds."
  (lisp-repl--remove-result-overlay)
  (save-excursion
    (goto-char where)
    (let* ((ov (make-overlay where where)))
      (overlay-put ov 'after-string
                   (propertize (format " => %s" value)
                               'face 'lisp-repl-result-face))
      (overlay-put ov 'lisp-repl-result t)
      (setq lisp-repl-overlay ov)
      (cond
       ((eq duration 'command)
        ;; Remove on next command
        (add-hook 'pre-command-hook #'lisp-repl--remove-result-overlay))
       ((numberp duration)
        ;; Remove after N seconds
        (run-at-time duration nil #'lisp-repl--remove-result-overlay)))))
  value)

(defun lisp-repl--get-repl-output ()
  "Get the last output from the inferior lisp buffer."
  (with-current-buffer "*inferior-lisp*"
    (save-excursion
      (goto-char (point-max))
      (forward-line -1)
      (let ((line (buffer-substring-no-properties
                   (line-beginning-position)
                   (line-end-position))))
        ;; Remove the prompt if present
        (if (string-match "^> " line)
            ""
          (string-trim line))))))

(defun lisp-repl-eval-last-sexp-inline ()
  "Evaluate the expression before point and show result inline."
  (interactive)
  (let ((eval-point (point)))
    ;; Do the normal evaluation
    (lisp-eval-last-sexp)
    ;; Wait a bit for output
    (sit-for 0.1)
    ;; Get and display the result
    (let ((result (lisp-repl--get-repl-output)))
      (when (and result (not (string-empty-p result)))
        (lisp-repl--make-result-overlay result eval-point 'command)))))

(defun lisp-repl-eval-defun-inline ()
  "Evaluate the current defun and show result inline."
  (interactive)
  (save-excursion
    (end-of-defun)
    (let ((end (point)))
      ;; Do the normal evaluation
      (lisp-eval-defun)
      ;; Wait a bit for output
      (sit-for 0.1)
      ;; Get and display the result
      (let ((result (lisp-repl--get-repl-output)))
        (when (and result (not (string-empty-p result)))
          (lisp-repl--make-result-overlay result end 'command))))))

(defun lisp-repl-eval-region-inline (start end)
  "Evaluate region from START to END and show result inline."
  (interactive "r")
  ;; Do the normal evaluation
  (lisp-eval-region start end)
  ;; Wait a bit for output
  (sit-for 0.1)
  ;; Get and display the result
  (let ((result (lisp-repl--get-repl-output)))
    (when (and result (not (string-empty-p result)))
      (lisp-repl--make-result-overlay result end 'command))))

;; Enhanced key bindings for inline evaluation
(define-key lisp-mode-map (kbd "C-x C-e") #'lisp-repl-eval-last-sexp-inline)
(define-key lisp-mode-map (kbd "C-M-x") #'lisp-repl-eval-defun-inline)
(define-key lisp-mode-map (kbd "C-c C-r") #'lisp-repl-eval-region-inline)
(define-key lisp-mode-map (kbd "C-c C-v") #'lisp-repl--remove-result-overlay)

(provide 'lisp-repl)
;;; lisp-repl.el ends here
