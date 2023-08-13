from flask import Flask, render_template, request
from demo import init_weight,predict
import numpy as np
app = Flask(__name__)
layer = init_weight()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    input1 = request.form.get('input1')
    input2 = request.form.get('input2')
    output = predict(layer,np.float32(input1),np.float32(input2))
    if output[0] == 0:
        output = "女"
    else:
        output = "男"
    return render_template('index.html',input1=input1,input2=input2,result=output)


if __name__ == '__main__':
    app.run(debug=True)