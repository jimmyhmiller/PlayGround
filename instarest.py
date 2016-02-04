from flask import Flask, make_response
from flask import jsonify
from flask.ext.pymongo import PyMongo
from flask import request
from bson.json_util import dumps
from bson.objectid import ObjectId
from functools import wraps
from flask.ext.cors import CORS



app = Flask(__name__)
app.config['MONGO_HOST'] = '192.168.99.102'
CORS(app)
mongo = PyMongo(app)

def sane_ids(coll):
    print(type(coll))
    print(isinstance(coll, list))
    if isinstance(coll, dict) and "_id" in coll:
        coll["id"] = str(coll["_id"])
        coll.pop("_id", None)
        return coll
    elif isinstance(coll, list):
        print("list!")
        return map(sane_ids, coll)
    else:
        return coll

@app.route("/<collection>", methods=["GET"])
def collections(collection):
    return dumps(sane_ids([x for x in mongo.db[collection].find()]))



@app.route("/<collection>/<id>", methods=["GET", "PUT"])
def gets(collection, id):
    entity = mongo.db[collection].find_one({"_id" : ObjectId(id)})
    if not entity:
        return ('entity not found', 404)
    return dumps(sane_ids(entity))



@app.route("/<collection>/<id>", methods=["PATCH", "PUT"])
def update(collection, id):
    mongo.db[collection].update({"_id": ObjectId(id)}, {"$set": request.get_json(True, True)})
    return gets(collection, id)


@app.route("/<collection>/<id>", methods=["DELETE"])
def delete(collection, id):
    mongo.db[collection].remove({"_id": ObjectId(id)})
    return ('',204)


@app.route("/<collection>", methods=["POST"])
def posts(collection):
    id = mongo.db[collection].insert(request.get_json(True, True))
    return gets(collection, id)

if __name__ == '__main__':
	app.run(debug=True)