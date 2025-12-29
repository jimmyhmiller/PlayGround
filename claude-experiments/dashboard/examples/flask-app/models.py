"""
Models for the Flask App
Simple in-memory models demonstrating the MVC pattern
"""


class Task:
    """A task/todo item that can be completed"""
    _counter = 0
    _all = []

    def __init__(self, title, done=False):
        Task._counter += 1
        self.id = Task._counter
        self.title = title
        self.done = done
        Task._all.append(self)

    @classmethod
    def all(cls):
        """Return all tasks"""
        return cls._all

    @classmethod
    def find(cls, id):
        """Find a task by ID"""
        return next((t for t in cls._all if t.id == id), None)

    @classmethod
    def pending(cls):
        """Return all incomplete tasks"""
        return [t for t in cls._all if not t.done]

    @classmethod
    def completed(cls):
        """Return all completed tasks"""
        return [t for t in cls._all if t.done]

    def toggle(self):
        """Toggle completion status"""
        self.done = not self.done
        return self


class Note:
    """A note with title and body text"""
    _counter = 0
    _all = []

    def __init__(self, title, body):
        Note._counter += 1
        self.id = Note._counter
        self.title = title
        self.body = body
        Note._all.append(self)

    @classmethod
    def all(cls):
        """Return all notes"""
        return cls._all

    @classmethod
    def find(cls, id):
        """Find a note by ID"""
        return next((n for n in cls._all if n.id == id), None)


# Seed initial data
Task("Learn Flask", done=True)
Task("Build a dashboard")
Task("Try Light Table concepts")

Note("Welcome", "This is a sample notes app built with Flask.")
Note("Flask Tips", "Use render_template() to render Jinja2 templates.")
