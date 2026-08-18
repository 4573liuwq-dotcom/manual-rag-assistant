from pymongo import MongoClient

# 1 获取mongodb连接
mongo_client = MongoClient("mongodb://192.168.200.139:27017")

# 2 创建数据
db = mongo_client["know_db"]

# 3 创建表
collection = db["users"]

# 1 添加数据
def insert_data():
    result = collection.insert_one({
        "name":"tom",
        "age":22,
        "gender":"male",
        "adress":"China"
    })
    print(result)

def insert_more_data():
    results =collection.insert_many(
        [
            {
                "name": "mary",
                "age": 10,
                "gender": "male"
            },
            {
                "name": "jack",
                "age": 28,
                "gender": "male"
            }

        ]
    )
    print(results)

# 查询
def query_data():
    for document in collection.find():
        print(document['name'])

# 条件查询
def query_condition():
    for document in collection.find({"name": "jack"}):
        print(document['name'],document['age'])

# 排序
# 1 升序  -1 降序
def query_sort():
    for document in collection.find().sort("age",-1).limit(1):
        print(document['name'],document['age'])

# 查询一条记录
def query_data():
    document = collection.find_one({"name": "jack"})
    print(document)

# update users set age = 20  where name='lucy'
# 修改
def update_data():
    result = collection.update_one(
        {"name": "jack"},
         {"$set":{"age": 30}}
    )
    print(result)

# 删除
def delete_data():
    result = collection.delete_one({"name": "jack"})
    print(result)

if __name__ == "__main__":
    print(db)
    print(collection)

    delete_data()
    # 修改
    # update_data()
    # 查询
    # query_sort()
    # query_condition()
    # query_data()
    # 添加
    # insert_data()
    # insert_more_data()