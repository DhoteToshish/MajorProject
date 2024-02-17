from flask import Flask, render_template
from data_processing import load_data
app = Flask(__name__, static_folder='static')

@app.route("/")
def home():
    data = load_data()
    return render_template("home.html", data=data)
    
if __name__ == "__main__":
    app.run(debug=True)