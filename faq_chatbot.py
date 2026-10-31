import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faqs = {
    "What is cybersecurity?":
        "Cybersecurity is the practice of protecting computers, networks and data from cyber attacks.",

    "What is a firewall?":
        "A firewall is a security system that monitors and controls incoming and outgoing network traffic.",

    "What is malware?":
        "Malware is malicious software designed to damage, steal or access computer systems.",

    "What is phishing?":
        "Phishing is a cyber attack where attackers trick users into sharing sensitive information.",

    "What is a virus?":
        "A computer virus is a malicious program that can infect files and spread between computers.",

    "What is antivirus?":
        "Antivirus is software that detects, prevents and removes malicious programs.",

    "What is hacking?":
        "Hacking is the process of identifying and exploiting weaknesses in computer systems.",

    "What is encryption?":
        "Encryption converts readable information into coded data to protect it from unauthorized access.",

    "What is a strong password?":
        "A strong password contains uppercase letters, lowercase letters, numbers and special characters.",

    "How to stay safe online?":
        "Use strong passwords, enable two-factor authentication and avoid suspicious links."
}

questions = list(faqs.keys())
answers = list(faqs.values())

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

faq_vectors = vectorizer.fit_transform(questions)


def chatbot():

    print("\n===================================")
    print("     CYBERSECURITY FAQ CHATBOT")
    print("===================================")

    print("\nAsk me anything about Cybersecurity!")
    print("Type 'exit' to close the chatbot.")

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("\nBot: Thank you! Stay Cyber Safe.")
            break

        if not user_input:
            print("Bot: Please enter a question.")
            continue

        user_vector = vectorizer.transform([user_input])

        similarity = cosine_similarity(
            user_vector, faq_vectors
        )

        best_match = similarity.argmax()
        confidence = similarity[0][best_match]

        if confidence >= 0.20:
            print("\nBot:", answers[best_match])
        else:
            print("\nBot: Sorry, I don't know the answer.")
            print("Please ask another cybersecurity question.")


if __name__ == "__main__":
    try:
        chatbot()
    except KeyboardInterrupt:
        print("\nChatbot closed.")