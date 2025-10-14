#ifndef SIMPLE_EVAL_H
#define SIMPLE_EVAL_H

// Simple evaluator that handles (+ num num) expressions
// Returns allocated string that caller must free
char* simple_eval(const char *code);

#endif
