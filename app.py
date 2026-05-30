from flask import Flask, render_template, request
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']

    df = pd.read_csv(file)

    sia = SentimentIntensityAnalyzer()

    sentiments = []

    for review in df['review']:
        score = sia.polarity_scores(str(review))

        if score['compound'] >= 0.05:
            sentiments.append('Positive')
        elif score['compound'] <= -0.05:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')

    df['Sentiment'] = sentiments

    positive = sentiments.count('Positive')
    negative = sentiments.count('Negative')
    neutral = sentiments.count('Neutral')

    return render_template(
        'result.html',
        positive=positive,
        negative=negative,
        neutral=neutral
    )

if __name__ == '__main__':
    app.run(debug=True)
