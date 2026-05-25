from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# FAQ Database
faqs = {
    "cancel order": "Yes, you can modify or cancel within 2 hours.",
    "international shipping": "Yes, we ship to over 100 countries.",
    "track order": "You will receive an email with a tracking link once shipped.",
    "return ": "We offer a 30-day money-back guarantee.",
    "warranty": "All products come with a 1-year manufacturer warranty.",
    "payment methods": "We accept Credit Cards, PayPal, and Apple Pay.",
    "is payment secure": "Yes, all transactions are encrypted.",
    "change address" :" yes , can change address of order before shipping ." ,
    "home delivery": "yes , we provide homedelivery all over the world ." ,
    "custumer support ": "please , dial 98375xxx for guided custumer support .",
    "hii": "hellow how can i help you ." ,
    "parcel": "yes we deliver the parcel ." ,

}

# NLP Logic
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(faqs.keys())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def chat():
    msg = request.form['msg']
    sims = cosine_similarity(vectorizer.transform([msg.lower()]), tfidf)
    best = sims.argmax()

    if sims[0][best] > 0.2:
        return list(faqs.values())[best]
    else:
        return "Sorry, I don't know."

if __name__ == '__main__':
    app.run(debug=True)
