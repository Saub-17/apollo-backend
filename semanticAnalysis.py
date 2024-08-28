from transformers import pipeline

# Load the emotion classifier
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

def analyze_sentiment(text):
    # Get the sentiment analysis results
    result = emotion_classifier(text)
    
    # Find the dominant sentiment with the highest score
    dominant_sentiment = max(result, key=lambda x: x['score'])
    
    return dominant_sentiment  # Return a single object representing the dominant sentiment
