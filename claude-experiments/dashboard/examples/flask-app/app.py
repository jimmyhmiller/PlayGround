"""
Flask App - Controllers/Routes
"""

from flask import Flask, render_template, request, redirect, url_for
from models import Task, Note

app = Flask(__name__)


@app.route('/')
def home():
    """Home page with overview"""
    return render_template('home.html',
        task_count=len(Task.all()),
        pending_count=len(Task.pending()),
        note_count=len(Note.all()))


@app.route('/tasks')
def task_list():
    """List all tasks"""
    return render_template('tasks/index.html',
        tasks=Task.all(),
        pending=Task.pending(),
        completed=Task.completed())


@app.route('/tasks/new')
def task_new():
    """Form to create a new task"""
    return render_template('tasks/new.html')


@app.route('/tasks', methods=['POST'])
def task_create():
    """Create a new task"""
    title = request.form.get('title', '').strip()
    if title:
        Task(title)
    return redirect(url_for('task_list'))


@app.route('/tasks/<int:id>/toggle', methods=['POST'])
def task_toggle(id):
    """Toggle task completion status"""
    task = Task.find(id)
    if task:
        task.toggle()
    return redirect(url_for('task_list'))


@app.route('/notes')
def note_list():
    """List all notes"""
    return render_template('notes/index.html', notes=Note.all())


@app.route('/notes/<int:id>')
def note_show(id):
    """Show a single note"""
    note = Note.find(id)
    if not note:
        return render_template('error.html', message="Note not found"), 404
    return render_template('notes/show.html', note=note)


@app.route('/notes/new')
def note_new():
    """Form to create a new note"""
    return render_template('notes/new.html')


@app.route('/notes', methods=['POST'])
def note_create():
    """Create a new note"""
    title = request.form.get('title', '').strip()
    body = request.form.get('body', '').strip()
    if title:
        Note(title, body)
    return redirect(url_for('note_list'))


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, port=5001)
