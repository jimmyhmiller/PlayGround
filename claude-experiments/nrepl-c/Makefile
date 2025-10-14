CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g
LDFLAGS = -luuid

TARGET = nrepl-server
SOURCES = server.c nrepl.c bencode.c simple_eval.c
OBJECTS = $(SOURCES:.c=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJECTS)

run: $(TARGET)
	./$(TARGET)
