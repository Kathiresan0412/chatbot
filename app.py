from flask import Flask, render_template, request, jsonify
from chat import get_response
from flask import Flask, redirect, session, flash
from flask_mysqldb import MySQL
import bcrypt
app = Flask(__name__)

app.secret_key = "your_secret_key"
# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '123456'
app.config['MYSQL_DB'] = 'game'
mysql = MySQL(app)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/signup', methods=['POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        # Hash the password
        hashed_password = bcrypt.hashpw(
            password.encode('utf-8'), bcrypt.gensalt())
        # Insert user into database
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                    (name, email, hashed_password))
        mysql.connection.commit()
        cur.close()
        flash("Signup successful", "success")
        return redirect('/')


@app.route('/signin', methods=['POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, password FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        if user and bcrypt.checkpw(password.encode('utf-8'), user[1].encode('utf-8')):
            session['user'] = user[0]
            flash("Login successful", "success")
            return redirect('/dashboard')
        else:
            flash("Invalid email or password", "danger")
            return redirect('/')


@app.get("/dashboard")
def index_get():
    return render_template("index.html")


@app.get("/ex")
def index_getex():
    return render_template("base.html")


@app.route('/blog')
def block():
    return render_template('blog.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/game-single')
def gameSingle():
    return render_template('game-single.html')


@app.route('/games')
def games():
    return render_template('games.html')


@app.route('/review')
def review():
    return render_template('review.html')


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == "__main__":
    app.run(debug=True)
