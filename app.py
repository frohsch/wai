from flask import Flask, request, render_template

# Import main library
import numpy as np

# Import Flask modules

# Import pickle to save our regression model
import pickle

# Initialize Flask and set the template folder to "template"
app = Flask(__name__, template_folder='template')

# create our "home" route using the "index.html" page
@app.route('/')
def home():
    return render_template('index.html')


"""RandomForest: model4.pkl, Linear Regression: model2.pkl"""
@app.route('/', methods=['POST'])
def predict():
    # model_F = [y for y in request.form.values()]
    # model_F=model_F[0:1]
    # model_F=[str(y)for y in model_F]
    #
    # if(model_F=='LinearRegression'):
    #     model = pickle.load(open('model2.pkl', 'rb'))
    # if (model_F == 'RandomForest'):
    #     model = pickle.load(open('model4.pkl', 'rb'))

    if(request.form['options']=='LinearRegression'):
        model = pickle.load(open('model2.pkl', 'rb'))  #LinearRegression
        int_features = [x for x in request.form.values()]
        int_features = int_features[1:4]
        int_features = [int(x) for x in int_features]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        if output < 0:
            return render_template('index.html',
                                   prediction_text="values entered not reasonable")
        elif output >= 0:
            return render_template('index.html', prediction_text='Predicted Power Result is : ${}'.format(output))
    else:
        model = pickle.load(open('model4.pkl', 'rb'))
        int_features = [x for x in request.form.values()]
        int_features = int_features[1:4]
        int_features = [int(x) for x in int_features]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        if output < 0:
            return render_template('index.html',
                                   prediction_text="values entered not reasonable")
        elif output >= 0:
            return render_template('index.html', prediction_text='Predicted Power Result is : ${}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)