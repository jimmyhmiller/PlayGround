#define _DEFAULT_SOURCE
#include "namespace.h"
#include "bundle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/select.h>
#include <time.h>

typedef const char* (*GreetingFn)();

// Set terminal to non-blocking mode
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

void update_greeting_code(int choice) {
    const char *greetings[] = {
        "const char* greeting() {\n    return \"Hello, World!\";\n}",
        "const char* greeting() {\n    return \"Greetings, Earthlings!\";\n}",
        "const char* greeting() {\n    return \"Yo! What's up?\";\n}"
    };

    if (choice < 1 || choice > 3) return;

    FILE *f = fopen("greeting_bundle.c", "w");
    if (!f) return;

    fprintf(f, "#include \"namespace.h\"\n");
    fprintf(f, "#include <stdio.h>\n\n");
    fprintf(f, "%s\n\n", greetings[choice - 1]);
    fprintf(f, "void bundle_init(Namespace *ns) {\n");
    fprintf(f, "    Definition *def = namespace_lookup(ns, \"greeting\");\n");
    fprintf(f, "    if (def) {\n");
    fprintf(f, "        definition_update(def, greeting);\n");
    fprintf(f, "    } else {\n");
    fprintf(f, "        namespace_define(ns, \"greeting\", DEF_FUNCTION, greeting);\n");
    fprintf(f, "    }\n");
    fprintf(f, "}\n");
    fclose(f);

    system("make greeting_bundle.so > /dev/null 2>&1");
    printf("\n[Rebuilt bundle with greeting %d]\n", choice);
}

int main() {
    Namespace *ns = namespace_create("main");
    Bundle *bundle = bundle_load("./greeting_bundle.so", ns);

    if (!bundle) {
        fprintf(stderr, "Failed to load greeting bundle\n");
        return 1;
    }

    set_nonblocking_input();

    printf("Greeting Demo - Press 1, 2, or 3 to change greeting (Ctrl+C to exit)\n");
    printf("1: Hello, World!\n");
    printf("2: Greetings, Earthlings!\n");
    printf("3: Yo! What's up?\n\n");

    time_t last_print = 0;

    while (1) {
        // Check for key press
        char c;
        if (read(STDIN_FILENO, &c, 1) > 0) {
            if (c >= '1' && c <= '3') {
                update_greeting_code(c - '0');
                bundle_reload(bundle);
            }
        }

        // Print greeting every 3 seconds
        time_t now = time(NULL);
        if (now - last_print >= 3) {
            system("clear");
            printf("Greeting Demo - Press 1, 2, or 3 to change greeting (Ctrl+C to exit)\n");
            printf("1: Hello, World!\n");
            printf("2: Greetings, Earthlings!\n");
            printf("3: Yo! What's up?\n\n");
            printf("Current greeting:\n");

            Definition *def = namespace_lookup(ns, "greeting");
            if (def) {
                GreetingFn greet = (GreetingFn)definition_get(def);
                printf(">>> %s\n", greet());
            }

            last_print = now;
        }

        usleep(100000); // 100ms
    }

    restore_terminal();
    bundle_unload(bundle);
    namespace_destroy(ns);

    return 0;
}
