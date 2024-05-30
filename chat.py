from nltk_utils import bag_of_words, tokenize
from model import NeuralNet
import random
import json
import MySQLdb
import pyodbc
import torch
from flask import Flask, redirect, session, flash
from flask import Flask, render_template, request, redirect, session
from flask_mysqldb import MySQL
import bcrypt
app = Flask(__name__)


def get_tabledata(tag):
    try:
        cursor = mysql.connection.cursor()
        query = 'SELECT Description FROM MedicaDetails WHERE Id = %s'
        cursor.execute(query, (tag,))
        row = cursor.fetchone()
        return row['Description'] if row else None
    except MySQLdb.Error as e:
        return f"Error occurred: {e}"
    finally:
        cursor.close()


app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '123456'
app.config['MYSQL_DB'] = 'game'

mysql = MySQL(app)

def InsertData(UserWants):
    try:
        cursor = mysql.connection.cursor()
        query = 'INSERT INTO UserNeeds (UserNeeds) VALUES (%s)'
        cursor.execute(query, (UserWants,))
        mysql.connection.commit()
    except MySQLdb.Error as e:
        mysql.connection.rollback()
        return f"Error occurred: {e}"
    finally:
        cursor.close()

app = Flask(__name__)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "End Game"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "ContactUs":
                    # return "Test...!"
                    result1 = get_tabledata(1)
                    result2 = get_tabledata(2)
                    return result1+" "+result2
                elif tag == "services":
                    result1 = get_tabledata(3)
                    result2 = get_tabledata(4)
                    result3 = get_tabledata(5)
                    result4 = get_tabledata(6)
                    result5 = get_tabledata(7)
                    result6 = get_tabledata(8)
                    result7 = get_tabledata(9)
                    result8 = get_tabledata(10)
                    result9 = get_tabledata(11)
                    return result1+" "+result2+" "+result3+" "+result4+" "+result5+" "+result6+" "+result7+" "+result8+" "+result9
                elif tag == "Locations":
                    result1 = get_tabledata(12)
                    result2 = get_tabledata(13)
                  # print(result1)
                    return (result1+"/t"+result2)
                elif tag == "AboutUs":
                    result1 = get_tabledata(14)
                    result2 = get_tabledata(15)
                    return result1+" "+result2
                elif tag == "WorkingTime":
                    result1 = get_tabledata(16)
                    return result1
                elif tag == "OurPromises":
                    result1 = get_tabledata(17)
                    return result1
                return random.choice(intent['responses'])
        result = InsertData(msg)
    return result


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
