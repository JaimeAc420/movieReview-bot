from flask import Flask, request, jsonify
import mods.module as module

app = Flask(__name__)

@app.route('/analyze', methods=['GET','POST'])
def analyze_opinion():
    data = request.get_json()
    text = data.get('text')

    # Use your machine learning model to analyze the text and get a prediction
    prediction = module.predict_opinion(text)

    return jsonify({'prediction': prediction})
    
if __name__ == '__main__':
    app.run(debug=True)