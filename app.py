from flask import Flask, request, jsonify
from flask_cors import CORS  # To handle CORS issues from frontend

# Simplified chatbot response function
def get_chatbot_response(user_query: str) -> str:
    """
    Generates a chatbot response based on the user's query.
    This is a simplified, keyword-based version for immediate testing.
    """
    if not user_query.strip():
        return "Please type something so I can help you."

    user_query_lower = user_query.lower()

    if "shipping" in user_query_lower or "delivery" in user_query_lower:
        return "We offer standard, express, and overnight shipping options. How can I help you further?"
    elif "return" in user_query_lower or "refund" in user_query_lower:
        return "Our return policy allows returns within 30 days for a full refund. Do you have a specific item in mind?"
    elif "stock" in user_query_lower or "available" in user_query_lower:
        return "To check stock for a specific product, please provide the product name or ID."
    elif "track" in user_query_lower or "order" in user_query_lower:
        return "You can track your order using the tracking number provided in your shipping confirmation email."
    elif "payment" in user_query_lower or "methods" in user_query_lower:
        return "We accept Visa, MasterCard, American Express, PayPal, and Google Pay."
    elif "discount" in user_query_lower or "coupon" in user_query_lower:
        return "Yes, new customers get 10% off their first order with code WELCOME10."
    elif "hello" in user_query_lower or "hi" in user_query_lower:
        return "Hello! How can I assist you with your online shop needs today?"
    elif "thank you" in user_query_lower or "thanks" in user_query_lower:
        return "You're welcome! Is there anything else I can help you with?"
    elif "bye" in user_query_lower or "goodbye" in user_query_lower:
        return "Goodbye! Have a great day."
    else:
        return "I'm sorry, I don't have enough information to answer that. Could you please rephrase your question or provide more details?"

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "Flask chatbot backend is running"})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"response": "No message provided."}), 400

    chatbot_response = get_chatbot_response(user_message)
    return jsonify({"response": chatbot_response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
