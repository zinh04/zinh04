- 👋 Hi, I’m @zinh04
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
zinh04/zinh04 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from fileInteractions import exportConfig
import pickle

def train(fileLocation):
    config = exportConfig()
    input_length = config["input_length"]
    train = {
        "x" : [],
        "y" : []
    }
    with open(fileLocation, "r+") as file:
        datasets = file.read().splitlines()
        for set in datasets:
            set = set.split(",")
            set = [ int(e) for e in set ]
            for i in range(len(set) - input_length - 1):
                train["x"].append(set[i : i + input_length])
                train["y"].append(set[i + input_length])

        randomForest = RandomForestRegressor(n_estimators = config["trees_in_the_forest"])
        randomForest.fit(train["x"], train["y"])
        with open("models/rf.pkl", "wb") as file:
            pickle.dump(randomForest, file)
            print("Đã lưu model Random Forest!")

        svr = SVR(kernel = config["kernel"])
        svr.fit(train["x"], train["y"])
        with open("models/svr.pkl", "wb") as file:
            pickle.dump(svr, file)
            print("Đã lưu model SVR!")

        neural = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=1000)
        neural.fit(train["x"], train["y"])
        with open("models/nn.pkl", "wb") as file:
            pickle.dump(neural, file)
            print("Đã lưu model Neural Network!")

def test(testcase):
    config = exportConfig()
    input_length = config["input_length"]
    test = {
        "x" : [],
        "y" : []
    }

    for i in range(len(testcase) - input_length - 1):
        test["x"].append(testcase[i : i + input_length])
        test["y"].append(testcase[i + input_length])

    with open("models/svr.pkl", "rb") as file:
        svr = pickle.load(file)
    with open("models/rf.pkl", "rb") as file:
        rf = pickle.load(file)
    with open("models/nn.pkl", "rb") as file:
        nn = pickle.load(file)

    predict = {
        "rf" : rf.predict(test["x"]),
        "svr" :  svr.predict(test["x"]),
        "nn" : nn.predict(test["x"])
    }

    accuracy = {
        "rf" : r2_score(test["y"], predict["rf"]),
        "svr" : r2_score(test["y"], predict["svr"]),
        "nn" : r2_score(test["y"], predict["nn"])
    }
    print("Độ chính xác\n-----------")
    [print(f"{al} : {round(accuracy[al], 3)}") for al in accuracy]

def ktest(fileLocation, testcase):
    config = exportConfig()
    input_length = config["input_length"]
    test = {
        "x" : [],
        "y" : []
    }
    train = {
        "x" : [],
        "y" : []
    }

    for i in range(len(testcase) - input_length - 1):
        test["x"].append(testcase[i : i + input_length])
        test["y"].append(testcase[i + input_length])
    with open(fileLocation, "r+") as file:
        datasets = file.read().splitlines()
        for set in datasets:
            set = set.split(",")
            set = [int(e) for e in set]
            for i in range(len(set) - input_length - 1):
                train["x"].append(set[i : i + input_length])
                train["y"].append(set[i + input_length])

    accuracy = {}

    for kernel in ("linear", "poly", "rbf", "sigmoid"):
        svr = SVR(kernel = kernel)
        svr.fit(train["x"], train["y"])
        prediction = svr.predict(test["x"])
        accuracy[kernel] = round(r2_score(test["y"], prediction), 4)

    print("Độ chính xác\n-----------")
    [print(f"{kernel} : {accuracy[kernel]}") for kernel in accuracy]


def s_predict(inp, model):
    with open("models/" + model + ".pkl", "rb") as file:
            model = pickle.load(file)
            return model.predict(inp)

def predict(inp, ml):
    inp = [int(e) for e in inp]
    if ml == "svr":
        res = s_predict([inp], "svr")
    elif ml == "rf":
        res = s_predict([inp], "rf")
    elif ml == "nn":
        res = s_predict([inp], "nn")
    elif ml == "all":
        res = dict()
        for al in ("svr", "rf", "nn"):
            res[al] = s_predict([inp], al)
        print("Kết quả dự đoán:")
        [print(model, ":", round(res[model][0], 4), "->", "Tài" if res[model][0] > 10.5 else "Xỉu") for model in res]

        return True
    else:
        print("Thuật toán không tồn tại hoặc không được hỗ trợ. Gõ help để xem danh sách các thuật toán khả dụng")
        return False
    print(f"Kết quả dự đoán là: {round(res[0], 4)}.")
    if 10.25 < res[0] < 10.75:
        print("Bạn nên bỏ lượt này")
    elif res[0] > 10.75:
        print("Khả năng ra Tài là cao hơn")
    else:
        print("Khả năng ra Xỉu là cao hơn")

                 
