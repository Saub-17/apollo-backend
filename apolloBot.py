from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from semanticAnalysis import analyze_sentiment  # Updated import
import os

# Load environment variables
load_dotenv()

# Configure the Google Generative AI
genai.configure(api_key='AIzaSyAFkRE1O8NV1mw1SyHBke-e0Rv55Qgyb4s')
model = genai.GenerativeModel("gemini-pro")

app = Flask(__name__)
CORS(app)

@app.route('/gemini', methods=['POST'])
def gemini():
    data = request.get_json()
    message = data.get('message', '')
    chat_history = data.get('history', [])

    try:
        # Generate poem response
        response = model.start_chat(history=chat_history).send_message(f"write a poem on {message}")
        poem = ''.join([chunk.text for chunk in response])

        # Get only the dominant sentiment
        dominant_sentiment = analyze_sentiment(poem)

        # Send back the poem and dominant sentiment
        return jsonify({
            "poem": poem,
            "sentiment": dominant_sentiment  # Now sending only the dominant sentiment
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=8000)
