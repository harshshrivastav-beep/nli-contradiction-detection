from flask import Flask, render_template, request
from model import predict_relationship

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    premise = ""
    hypothesis = ""

    if request.method == "POST":
        premise = request.form["premise"]
        hypothesis = request.form["hypothesis"]
        result, confidence = predict_relationship(premise, hypothesis)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        premise=premise,
        hypothesis=hypothesis
    )

if __name__ == "__main__":
    app.run(debug=True)
