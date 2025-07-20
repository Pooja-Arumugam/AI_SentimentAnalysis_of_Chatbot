from transformers import pipeline
import re
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Load chat file
def load_chat_history(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# Optional: extract only user messages
def extract_user_messages(chat_text):
    # Assumes chat is in "User: ..." and "Bot: ..." format
    user_lines = re.findall(r'User:(.*)', chat_text, flags=re.IGNORECASE)
    return " ".join(user_lines).strip()

# Sentiment prediction
def predict_satisfaction(text):
    classifier = pipeline("sentiment-analysis")
    result = classifier(text)[0]
    label = result['label']
    score = result['score']

    if label == 'POSITIVE' and score > 0.7:
        return "Customer is likely SATISFIED ✅"
    else:
        return "Customer is likely UNSATISFIED ❌"

# Main
if __name__ == "__main__":
    filepath = 'ChatBot Feedback\Chat1.txt'  # Your file here
    chat = load_chat_history(filepath)
    user_text = extract_user_messages(chat)
    print("Analyzing sentiment...")
    outcome = predict_satisfaction(user_text)
    print(outcome)
