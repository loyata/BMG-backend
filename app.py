from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import mysql.connector
import socket

import random

import matplotlib
matplotlib.use('Agg')



app = Flask(__name__)  # See here

# mysql = MySQL(app)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/get_my_ip", methods=["GET"])
def get_my_ip():
    return jsonify({'ip': request.remote_addr}), 200


@app.route("/liver_graph", methods=["GET"])
def graph():
    cnx = mysql.connector.connect(
        host='frwahxxknm9kwy6c.cbetxkdyhwsb.us-east-1.rds.amazonaws.com',
        user='j6qbx3bgjysst4jr',
        password='mcbsdk2s27ldf37t',
        database='nkw2tiuvgv6ufu1z'
        # port=3306
    )
    # // host: "frwahxxknm9kwy6c.cbetxkdyhwsb.us-east-1.rds.amazonaws.com",
    #
    # // user: "j6qbx3bgjysst4jr",
    # // password: 'mcbsdk2s27ldf37t',
    # // database: 'evjygdytdp2ev0d',

    cursor = cnx.cursor()
    query = "SELECT age, gender, COUNT(*) FROM liver WHERE prediction='yes' GROUP BY age, gender"
    cursor.execute(query)
    results = cursor.fetchall()
    df = pd.DataFrame(results, columns=['age', 'gender', 'count'])
    df_pivot = df.pivot(index='age', columns='gender', values='count')
    fig, ax = plt.subplots()
    df_pivot.plot(kind='bar', ax=ax)
    ax.set_xlabel('Age')
    ax.set_ylabel('Count of "Prediction Yes"')
    ax.set_title('Prediction Yes by Age and Gender')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('ascii')
    response = img_base64
    # response.headers['Content-Type'] = 'text/html'
    return response


@app.route("/predict_liver_disease", methods=["POST"])
@cross_origin()
def predict_liver_disease():
    cnx = mysql.connector.connect(
        host='frwahxxknm9kwy6c.cbetxkdyhwsb.us-east-1.rds.amazonaws.com',
        user='j6qbx3bgjysst4jr',
        password='mcbsdk2s27ldf37t',
        database='nkw2tiuvgv6ufu1z'
    )
    data = request.json
    print(data)
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    try:
        input = [[
            float(data['age']),
            float(data['tb']),
            float(data['db']),
            float(data['ap']),
            float(data['aa1']),
            float(data['aa2']),
            float(data['tp']),
            float(data['al']),
            float(data['ag']),
            1 if data['gender'] == 'female' else 0,
            1 if data['gender'] == 'male' else 1
        ]]

        print(input)

        res = loaded_model.predict(input)[0]

        # cur = mysql.connection.cursor()
        cursor = cnx.cursor()




        query = "INSERT INTO liver (" \
                "name, " \
                "age, " \
                "gender, " \
                "total_bilirubin, " \
                "direct_bilirubin, " \
                "alkaline_phosphatase, " \
                "alanine_aminotransferase, " \
                "aspartate_aminotransferase, " \
                "total_proteins, " \
                "albumin, " \
                "albumin_and_globulin_ratio, " \
                "prediction) " \
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        values = (
        data['name'], data['age'], data['gender'], data['tb'], data['db'], data['ap'], data['aa1'], data['aa2'],
        data['tp'], data['al'], data['ag'], "yes" if res == 1 else "no")
        cursor.execute(query, values)
        cnx.commit()
        cursor.close()
        cnx.close()

        return "Potential liver disease detected, further examination required." if res == 1 else "No liver disease detected."
    except Exception as e:
        print("Error: {}".format(str(e)))
        return "input error"



@app.route("/predict_MS", methods=["POST"])
@cross_origin()
def predict_MS():
    MS_model = pickle.load(open('model', 'rb'))
    file = request.files['file']
    data = pd.read_csv(file)
    X = data
    # Make predictions on the test data
    y_pred = MS_model.predict(X)
    if (y_pred == 1):
        diagnose = 'patient suffers from MS'
    elif (y_pred == 0):
        diagnose = 'patient does not suffer from MS'
    print(diagnose)
    return diagnose


@app.route("/get_all_liver_data", methods=["GET"])
@cross_origin()
def get_all_liver_data():
    cnx = mysql.connector.connect(
        host='frwahxxknm9kwy6c.cbetxkdyhwsb.us-east-1.rds.amazonaws.com',
        user='j6qbx3bgjysst4jr',
        password='mcbsdk2s27ldf37t',
        database='nkw2tiuvgv6ufu1z'
    )
    cursor = cnx.cursor()
    query = "SELECT * FROM liver"
    cursor.execute(query)
    results = cursor.fetchall()
    return results

@app.route("/predict_MS_direct/<int:id>", methods=["GET"])
@cross_origin()
def predict_MS_direct(id):
    MS_model = pickle.load(open('model', 'rb'))
    cnx = mysql.connector.connect(
        host='frwahxxknm9kwy6c.cbetxkdyhwsb.us-east-1.rds.amazonaws.com',
        user='j6qbx3bgjysst4jr',
        password='mcbsdk2s27ldf37t',
        database='nkw2tiuvgv6ufu1z'
    )
    cursor = cnx.cursor()

    query = "SELECT pyramidal, cerebella, brain_stem, sensory, visual, mental, bowel_and_bladder_function, mobility " \
            "FROM physical_test_ms WHERE patient_id = %s"
    cursor.execute(query, (id,))

    data = cursor.fetchone()
    index = ["0"]
    columns = ['Pyramidal', 'Cerebella', 'Brain stem', 'Sensory', 'Visual', 'Mental', 'Bowel and bladder function',
               'Mobility']

    # Create a pandas DataFrame with the specified row and column names
    df = pd.DataFrame([data], index=index, columns=columns)
    X = df
    print(X)
    # Make predictions on the test data
    y_pred = MS_model.predict(X)
    if (y_pred == 1):
        diagnose = 'patient suffers from MS'
    elif (y_pred == 0):
        diagnose = 'patient does not suffer from MS'
    print(diagnose)
    return diagnose



@app.route("/predict_liver_disease_direct/<int:id>", methods=["GET"])
@cross_origin()
def predict_liver_disease_direct(id):
    cnx = mysql.connector.connect(
        host='frwahxxknm9kwy6c.cbetxkdyhwsb.us-east-1.rds.amazonaws.com',
        user='j6qbx3bgjysst4jr',
        password='mcbsdk2s27ldf37t',
        database='nkw2tiuvgv6ufu1z'
    )
    cursor = cnx.cursor()
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    query = "SELECT name, age, gender, total_bilirubin, direct_bilirubin, " \
            "alkaline_phosphatase, alanine_aminotransferase, aspartate_aminotransferase, " \
            "total_proteins, albumin, albumin_and_globulin_ratio " \
            "FROM liver " \
            "WHERE patient_id = %s"
    cursor.execute(query, (id,))

    result = cursor.fetchone()

    input = [[
        float(result[1]),
        float(result[3]),
        float(result[4]),
        float(result[5]),
        float(result[6]),
        float(result[7]),
        float(result[8]),
        float(result[9]),
        float(result[10]),
        1 if result[2] == 'female' else 0,
        1 if result[2] == 'male' else 1
    ]]
    res = loaded_model.predict(input)[0]
    return "Potential liver disease detected, further examination required." if res == 1 else "No liver disease detected."




if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    app.run(debug=True, port=port)
