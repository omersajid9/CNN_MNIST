from import_list import *

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/digit_prediction", methods=["POST"])

def digit_prediction():
    if (request.method == "POST"):
        print("HERE")
        img = request.get_json()
        pickle.dump(img, open("trim.png", "wb"))
        arc = pickle.load(open("final_network.py", "rb"))
        img = np.asarray(img)
        img = 1 - img / 255.
        img = img.reshape(1, 1, 28, 28)
        (output, pred) = arc.predict(img)
        print("H")
        print(int(round(pred*100, 3)))
        data = {"digit": int(output), "pred": int(round(pred*100, 3))}
        return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)