from vial import Vial
app = Vial(name="awesome-app", host="192.168.99.100:9092")

@app.event("hello")
def hello(message):
    return "hello world"

if __name__ == '__main__':
    app.run()
