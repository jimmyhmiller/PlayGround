from flask import Flask
from flask import jsonify
from flask.ext.pymongo import PyMongo
from flask import request
from bson.json_util import dumps
from bson.objectid import ObjectId

app = Flask(__name__)
mongo = PyMongo(app)

@app.route("/<collection>/", methods=["GET"])
def Collections(collection):
    return dumps(mongo.db[collection].find())


@app.route("/<collection>/<id>", methods=["GET", "PUT"])
def Gets(collection, id):
    return  dumps(mongo.db[collection].find({"_id" : ObjectId(id)}))


@app.route("/<collection>/", methods=["POST"])
def Posts(collection):
    return dumps(mongo.db[collection].insert(request.get_json(True, True)))

if __name__ == '__main__':
	app.run(debug=True)