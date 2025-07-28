import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import torch.nn.functional as F

# --- 1. Data Preparation ---
# In a real scenario, you would load your online shop's Q&A data.
# Example: A CSV file with columns like 'question' and 'answer'.
# For demonstration, let's create a dummy dataset.
data = {
    'question': [
        "What are your shipping options?",
        "How long does delivery take?",
        "Can I return an item?",
        "What is your return policy?",
        "Do you have this product in stock?",
        "How can I track my order?",
        "What payment methods do you accept?",
        "Is there a discount for first-time buyers?",
        "Tell me about product X.",
        "Where is my package?",
        "Can I change my shipping address?",
        "Do you offer international shipping?",
        "What is the warranty on this item?",
        "How do I apply a coupon code?",
        "Can I cancel my order?"
    ],
    'answer': [
        "We offer standard, express, and overnight shipping options. Details are available on our shipping policy page.",
        "Standard delivery usually takes 3-5 business days. Express is 1-2 days, and overnight is next business day.",
        "Yes, you can return most items within 30 days of purchase. Please see our return policy for details.",
        "Our return policy allows returns within 30 days for a full refund, provided the item is in its original condition.",
        "To check stock for a specific product, please provide the product name or ID.",
        "You can track your order using the tracking number provided in your shipping confirmation email.",
        "We accept Visa, MasterCard, American Express, PayPal, and Google Pay.",
        "Yes, new customers get 10% off their first order with code WELCOME10.",
        "Product X is a high-quality, durable item designed for [specific use case]. It comes with a 1-year warranty.",
        "Please provide your order number and I can check the status of your package for you.",
        "Unfortunately, we cannot change the shipping address once an order has been placed. Please ensure your address is correct at checkout.",
        "Yes, we offer international shipping to many countries. Shipping costs and times vary by destination.",
        "Most items come with a standard 1-year manufacturer's warranty. Extended warranties may be available for purchase.",
        "You can apply a coupon code during the checkout process in the designated 'Promo Code' box.",
        "Orders can be canceled within 24 hours of placement, provided they have not yet been shipped."
    ]
}

df = pd.DataFrame(data)

# For a real chatbot, you'd likely use a more sophisticated approach
# like intent classification + slot filling, or a seq2seq model for generation.
# For simplicity, we'll treat this as a classification problem where we map
# a question to its most similar answer from a predefined set.

# Create unique labels for each answer
df['label'] = df['answer'].astype('category').cat.codes
label_to_answer = {i: answer for i, answer in enumerate(df['answer'].unique())}

# Split data (optional for small datasets, but good practice)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# --- 2. Model Loading and Tokenization ---
MODEL_NAME = "distilbert-base-uncased" # A smaller, faster BERT model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(df['label'].unique()))

# Prepare dataset for training
class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_encodings = tokenizer(list(train_df['question']), truncation=True, padding=True)
val_encodings = tokenizer(list(val_df['question']), truncation=True, padding=True)

train_dataset = ChatDataset(train_encodings, list(train_df['label']))
val_dataset = ChatDataset(val_encodings, list(val_df['label']))

# --- 3. Fine-tuning the Model (Conceptual) ---
# This part would typically be run once to train your model.
# For a real application, you'd save the trained model.

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,   # batch size per device during training
    per_device_eval_batch_size=8,    # batch size per device during evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # Evaluate every epoch
    save_strategy="epoch",           # Save every epoch
    load_best_model_at_end=True,     # Load the best model at the end of training
    metric_for_best_model="eval_loss", # Metric to monitor for best model
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
)

# trainer.train() # Uncomment to train the model
# trainer.save_model("./fine_tuned_chatbot_model") # Uncomment to save the model

# --- 4. Chatbot Inference Function ---
# This function would be exposed via an API (e.g., Flask, FastAPI)
# to handle incoming user queries.

# Load the fine-tuned model and tokenizer (if you saved it)
# model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_chatbot_model")
# tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_chatbot_model")

def get_chatbot_response(user_query: str) -> str:
    """
    Generates a chatbot response based on the user's query.
    In a real system, this would involve loading the fine-tuned model
    and using it for inference.
    """
    if not user_query.strip():
        return "Please type something so I can help you."

    # For demonstration, we'll simulate a response based on keywords
    # In a real scenario, you'd use the trained model to predict the best answer.

    user_query_lower = user_query.lower()

    # Simple keyword-based matching for demonstration
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
        # If using the actual model:
        # inputs = tokenizer(user_query, return_tensors="pt", truncation=True, padding=True)
        # with torch.no_grad():
        #     outputs = model(**inputs)
        # logits = outputs.logits
        # predicted_label_id = torch.argmax(logits, dim=1).item()
        # return label_to_answer.get(predicted_label_id, "I'm sorry, I don't have enough information to answer that. Could you please rephrase your question or provide more details?")
        return "I'm sorry, I don't have enough information to answer that. Could you please rephrase your question or provide more details?"

# Example of how to use the function:
# if __name__ == "__main__":
#     print(get_chatbot_response("How do I return something?"))
#     print(get_chatbot_response("Where is my order?"))
#     print(get_chatbot_response("Do you have any discounts?"))
#     print(get_chatbot_response("What is the weather like?")) # Out of scope