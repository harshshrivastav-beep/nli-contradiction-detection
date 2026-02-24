from flask import Flask, render_template, request
from model import predict_relationship

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    premise = ""
    hypothesis = ""

    if request.method == "POST":
        premise = request.form["premise"]
        hypothesis = request.form["hypothesis"]
        result = predict_relationship(premise, hypothesis)

    return render_template(
        "index.html",
        result=result,
        premise=premise,
        hypothesis=hypothesis
    )

if __name__ == "__main__":
    app.run(debug=True)
