from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from gpt2_script import generate_response

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Enter your Question."}), 400

        answer = generate_response(question)

        answer = answer.replace(question, '').strip()

        return jsonify({"question": question, "answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Server run: http://127.0.0.1:5000/")
    app.run(debug=True)
