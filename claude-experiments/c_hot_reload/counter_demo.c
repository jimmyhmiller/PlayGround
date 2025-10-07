#define _DEFAULT_SOURCE
#include "namespace.h"
#include "bundle.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <time.h>

typedef int (*ChangeFn)(int);

void set_nonblocking_input() {
    struct termios ttystate;
    tcgetattr(STDIN_FILENO, &ttystate);
    ttystate.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &ttystate);
    fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
}

void restore_terminal() {
    struct termios ttystate;
    tcgetattr(STDIN_FILENO, &ttystate);
    ttystate.c_lflag |= (ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &ttystate);
}

void update_change_function(int choice) {
    const char *implementations[] = {
        "int change(int current) {\n    return current + 1;  // Increment by 1\n}",
        "int change(int current) {\n    return current + 10;  // Increment by 10\n}",
        "int change(int current) {\n    return current * 2;  // Double\n}",
        "int change(int current) {\n    return current - 5;  // Decrement by 5\n}",
        "int change(int current) {\n    return 0;  // Reset to 0\n}"
    };

    if (choice < 1 || choice > 5) return;

    FILE *f = fopen("counter_bundle.c", "w");
    if (!f) return;

    fprintf(f, "#include \"namespace.h\"\n\n");
    fprintf(f, "%s\n\n", implementations[choice - 1]);
    fprintf(f, "void bundle_init(Namespace *ns) {\n");
    fprintf(f, "    Definition *def = namespace_lookup(ns, \"change\");\n");
    fprintf(f, "    if (def) {\n");
    fprintf(f, "        definition_update(def, change);\n");
    fprintf(f, "    } else {\n");
    fprintf(f, "        namespace_define(ns, \"change\", DEF_FUNCTION, change);\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n");
    fclose(f);

    system("make counter_bundle.so > /dev/null 2>&1");
    printf("\n[Switched to change function %d]\n", choice);
}

int main() {
    Namespace *ns = namespace_create("main");
    Bundle *bundle = bundle_load("./counter_bundle.so", ns);

    if (!bundle) {
        fprintf(stderr, "Failed to load counter bundle\n");
        return 1;
    }

    set_nonblocking_input();

    int count = 0;
    time_t last_update = 0;

    printf("Counter Demo - Press 1-5 to change behavior (Ctrl+C to exit)\n");
    printf("1: +1    2: +10    3: *2    4: -5    5: Reset to 0\n\n");

    while (1) {
        // Check for key press
        char c;
        if (read(STDIN_FILENO, &c, 1) > 0) {
            if (c >= '1' && c <= '5') {
                update_change_function(c - '0');
                bundle_reload(bundle);
            }
        }

        // Update count every second
        time_t now = time(NULL);
        if (now - last_update >= 1) {
            system("clear");
            printf("Counter Demo - Press 1-5 to change behavior (Ctrl+C to exit)\n");
            printf("1: +1    2: +10    3: *2    4: -5    5: Reset to 0\n\n");

            Definition *def = namespace_lookup(ns, "change");
            if (def) {
                ChangeFn change = (ChangeFn)definition_get(def);
                count = change(count);
                printf("Current count: %d\n", count);
            }

            last_update = now;
        }

        usleep(100000); // 100ms
    }

    restore_terminal();
    bundle_unload(bundle);
    namespace_destroy(ns);

    return 0;
}
