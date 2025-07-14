from flask import Flask, render_template, request
import re
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download("stopwords")

app = Flask(__name__)

# --- All your provided samples ---
bully_samples = [
    "You're so stupid and ugly", "Nobody wants you around", "I hate you",
    "Why don't you just disappear?", "Everyone hates you", "You are a total failure",
    "You’re so fat it’s disgusting", "Go kill yourself", "You're such a loser",
    "Stop pretending to be important", "You're annoying and useless",
    "Nobody cares about your opinion", "You're a joke", "Shut up already",
    "No one likes you anyway", "You look like trash", "You're a pathetic excuse",
    "Get lost freak", "I hope you fall off a cliff", "This is why nobody loves you",
    "You're the dumbest person I've ever met", "Why are you even alive?",
    "You're a pathetic joke", "You ruin everything", "Go cry somewhere else",
    "You're nothing but a failure", "I hate everything about you", "You deserve all the hate you get",
    "Shut up, no one asked you", "Stop embarrassing yourself", "Your voice is annoying",
    "Just stop talking forever", "You're a complete idiot", "You're not even funny, just sad",
    "You're worse than useless", "That outfit makes you look dumb", "You're trash at this game",
    "You’ll never be good enough", "No one wants to be your friend", "Just quit already",
    "You're the reason everything fails", "Disgusting person", "I want to fuck you",
    "Shut up and die", "You're a dumb bitch", "You’re a pathetic loser",
    "No one gives a damn about you", "You're annoying af", "Go fuck yourself",
    "You're ugly and disgusting", "Stop acting like a slut"
]

non_bully_samples = [
    "You're doing amazing, keep going 💪", "Let’s catch up soon!", "Congratulations on your win!",
    "That’s such a beautiful photo!", "I'm here if you need anything ❤️", "You are really talented!",
    "Thanks for helping me today 😊", "Wishing you a fantastic day!", "You inspire me every day",
    "What a wonderful idea!", "You're so thoughtful", "Great effort on the project",
    "That’s impressive work!", "You made my day better", "I appreciate your support",
    "Thanks for being kind", "You’re very intelligent", "That’s really generous of you",
    "You’re a wonderful friend", "That was really brave of you", "Always love your positivity",
    "I admire your honesty", "I’m glad you’re in my life", "Keep smiling, you're amazing 😄",
    "Such a kind message!", "You're the best!", "Well done on your progress!",
    "Proud of what you've achieved", "This is very inspiring", "Nice game, well played",
    "You are amazing!", "You're doing great", "Have a nice day",
    "That was really helpful, thank you!", "Keep up the good work",
    "I love your creativity", "What a beautiful message",
    "This made my day ❤️", "You’re such a kind soul",
    "Awesome job!", "Well done on your success",
    "Let’s meet for coffee soon!", "Thanks for your support",
    "You are so sweet", "Wishing you the best always",
    "Proud of you!", "That was thoughtful", "You inspire me",
    "Thanks again for helping me", "You’re truly a good friend"
]

# --- Text Preprocessing ---
tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = tokenizer.tokenize(text)
    return " ".join([w for w in tokens if w not in stop_words])

# Dataset
X = bully_samples + non_bully_samples
y = ["bullying"] * len(bully_samples) + ["not bullying"] * len(non_bully_samples)
X_clean = [clean_text(text) for text in X]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression())
])
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
eval_report = classification_report(y_test, y_pred)

# Confidence threshold & prediction
bully_index = list(model.named_steps["clf"].classes_).index("bullying")
def predict_tweet(tweet):
    cleaned = clean_text(tweet)
    proba = model.predict_proba([cleaned])[0][bully_index]
    label = "bullying" if proba >= 0.50 else "not bullying"
    return label, round(proba * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    show_eval = False
    tweet_text = ""

    if request.method == "POST":
        tweet_text = request.form.get("tweet", "")
        if "evaluation" in request.form:
            show_eval = True
        elif tweet_text.strip():
            prediction, confidence = predict_tweet(tweet_text)

    return render_template("index.html", prediction=prediction, confidence=confidence,
                           eval_report=eval_report if show_eval else None)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

