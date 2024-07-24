# from flask_pymongo import PyMongo
# from flask_bcrypt import Bcrypt

# mongo = PyMongo()
# bcrypt = Bcrypt()

# def init_db(app):
#     mongo.init_app(app)
#     bcrypt.init_app(app)

# def create_user(first_name, password):
#     hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
#     user = {
#         "first_name": first_name,
#         "password": hashed_password
#     }
#     return mongo.db.users.insert_one(user)

# def find_user_by_first_name(first_name):
#     return mongo.db.users.find_one({"first_name": first_name})
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
import gridfs
from bson.objectid import ObjectId

mongo = PyMongo()
bcrypt = Bcrypt()

def init_db(app):
    mongo.init_app(app)
    bcrypt.init_app(app)

def create_user(first_name, password):
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user = {
        "first_name": first_name,
        "password": hashed_password
    }
    return mongo.db.users.insert_one(user)

def find_user_by_first_name(first_name):
    return mongo.db.users.find_one({"first_name": first_name})

def get_user_collection(user_id):
    return mongo.db.get_collection(str(user_id))

def save_file(file, file_type, user_id):
    fs = gridfs.GridFS(mongo.db)
    file_id = fs.put(file, filename=f"{file_type}.png", user_id=ObjectId(user_id))
    return file_id

def get_file(file_type, user_id):
    fs = gridfs.GridFS(mongo.db)
    file = fs.find_one({"filename": f"{file_type}.png", "user_id": ObjectId(user_id)})
    return file
