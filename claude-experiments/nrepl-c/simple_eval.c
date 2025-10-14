#define _POSIX_C_SOURCE 200809L
#include "simple_eval.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

// Simple parser for (+ num num) expressions
char* simple_eval(const char *code) {
    if (!code) return NULL;

    // Skip whitespace
    while (*code && isspace(*code)) code++;

    // Check for (+ num num) pattern
    if (*code != '(') {
        return strdup("nil");
    }
    code++; // skip '('

    // Skip whitespace
    while (*code && isspace(*code)) code++;

    // Check for + operator
    if (*code != '+') {
        return strdup("nil");
    }
    code++; // skip '+'

    // Parse first number
    while (*code && isspace(*code)) code++;
    int num1 = 0;
    int negative1 = 0;
    if (*code == '-') {
        negative1 = 1;
        code++;
    }
    while (*code && isdigit(*code)) {
        num1 = num1 * 10 + (*code - '0');
        code++;
    }
    if (negative1) num1 = -num1;

    // Parse second number
    while (*code && isspace(*code)) code++;
    int num2 = 0;
    int negative2 = 0;
    if (*code == '-') {
        negative2 = 1;
        code++;
    }
    while (*code && isdigit(*code)) {
        num2 = num2 * 10 + (*code - '0');
        code++;
    }
    if (negative2) num2 = -num2;

    // Skip whitespace
    while (*code && isspace(*code)) code++;

    // Check for closing paren
    if (*code != ')') {
        return strdup("nil");
    }

    // Calculate result
    int result = num1 + num2;

    // Convert to string
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%d", result);

    return strdup(buffer);
}
