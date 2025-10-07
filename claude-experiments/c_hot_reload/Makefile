CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g -fPIC
LDFLAGS = -ldl -lpthread -rdynamic

# Main executables
TARGET = hot_reload
GREETING_DEMO = greeting_demo
COUNTER_DEMO = counter_demo
OBJS = main.o namespace.o bundle.o
GREETING_OBJS = greeting_demo.o namespace.o bundle.o
COUNTER_OBJS = counter_demo.o namespace.o bundle.o

# Test executables
TEST_NAMESPACE = test_namespace
TEST_BUNDLE = test_bundle
TEST_CROSS = test_cross_bundle
TEST_NAMESPACE_OBJS = test_namespace.o namespace.o
TEST_BUNDLE_OBJS = test_bundle.o namespace.o bundle.o
TEST_CROSS_OBJS = test_cross_bundle.o namespace.o bundle.o

# Bundles
BUNDLES = example_bundle.so util_bundle.so dependent_bundle.so greeting_bundle.so counter_bundle.so
TEST_BUNDLES = test_bundle_v1.so test_bundle_v2.so

all: $(TARGET) $(GREETING_DEMO) $(COUNTER_DEMO) $(BUNDLES)

tests: $(TEST_NAMESPACE) $(TEST_BUNDLE) $(TEST_CROSS) $(BUNDLES) $(TEST_BUNDLES)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

$(GREETING_DEMO): $(GREETING_OBJS)
	$(CC) $(GREETING_OBJS) -o $(GREETING_DEMO) $(LDFLAGS)

$(COUNTER_DEMO): $(COUNTER_OBJS)
	$(CC) $(COUNTER_OBJS) -o $(COUNTER_DEMO) $(LDFLAGS)

$(TEST_NAMESPACE): $(TEST_NAMESPACE_OBJS)
	$(CC) $(TEST_NAMESPACE_OBJS) -o $(TEST_NAMESPACE) $(LDFLAGS)

$(TEST_BUNDLE): $(TEST_BUNDLE_OBJS)
	$(CC) $(TEST_BUNDLE_OBJS) -o $(TEST_BUNDLE) $(LDFLAGS)

$(TEST_CROSS): $(TEST_CROSS_OBJS)
	$(CC) $(TEST_CROSS_OBJS) -o $(TEST_CROSS) $(LDFLAGS)

%.so: %.c namespace.h
	$(CC) $(CFLAGS) -shared $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(GREETING_DEMO) $(COUNTER_DEMO) $(OBJS) $(GREETING_OBJS) $(COUNTER_OBJS) $(BUNDLES)
	rm -f $(TEST_NAMESPACE) $(TEST_BUNDLE) $(TEST_CROSS) $(TEST_NAMESPACE_OBJS) $(TEST_BUNDLE_OBJS) $(TEST_CROSS_OBJS) $(TEST_BUNDLES)
	rm -f test_bundle_reload.c test_bundle_reload.so

rebuild-bundle: $(BUNDLES)

run-tests: tests
	@echo "Running namespace tests..."
	@./$(TEST_NAMESPACE)
	@echo ""
	@echo "Running bundle tests..."
	@./$(TEST_BUNDLE)
	@echo ""
	@echo "Running cross-bundle tests..."
	@./$(TEST_CROSS)

.PHONY: all clean rebuild-bundle tests run-tests
