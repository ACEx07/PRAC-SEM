import numpy as np
from collections import defaultdict

emails = [
    ('Buy now, limited offer!', 1),
    ('Hello, how are you?', 0),
    ('Exclusive deal just for you', 1),
    ('Meeting agenda for next week', 0),
    ('Claim your prize now', 1)
]

def tokenize(text):
    return text.lower().split()

word_count_spam = defaultdict(int)
word_count_ham = defaultdict(int)
total_spam = 0
total_ham = 0
vocabulary = set()

for email, label in emails:
    words = tokenize(email)
    vocabulary.update(words)
    if label == 1:
        for word in words:
            word_count_spam[word] += 1
        total_spam += len(words)
    else:
        for word in words:
            word_count_ham[word] += 1
        total_ham += len(words)

def calculate_word_probabilities(word_counts, total_count, vocab_size, alpha=1.0):
    probabilities = defaultdict(lambda: alpha / (total_count + alpha * vocab_size))
    for word, count in word_counts.items():
        probabilities[word] = (count + alpha) / (total_count + alpha * vocab_size)
    return probabilities

vocab_size = len(vocabulary)
prob_word_spam = calculate_word_probabilities(word_count_spam, total_spam, vocab_size)
prob_word_ham = calculate_word_probabilities(word_count_ham, total_ham, vocab_size)

p_spam = sum(1 for _, label in emails if label == 1) / len(emails)
p_ham = 1 - p_spam

def predict_spam(email):
    words = tokenize(email)
    log_p_spam_given_email = np.log(p_spam)
    log_p_ham_given_email = np.log(p_ham)

    for word in words:
        log_p_spam_given_email += np.log(prob_word_spam[word])
        log_p_ham_given_email += np.log(prob_word_ham[word])

    max_log_prob = max(log_p_spam_given_email, log_p_ham_given_email)
    p_spam_given_email = np.exp(log_p_spam_given_email - max_log_prob)
    p_ham_given_email = np.exp(log_p_ham_given_email - max_log_prob)

    return p_spam_given_email / (p_spam_given_email + p_ham_given_email)

test_emails = [
    'Limited offer, claim your prize now!',
    'Hello, how about meeting for lunch tomorrow?',
    'Exclusive deal just for you'
]

for email in test_emails:
    spam_probability = predict_spam(email)
    print(f"Email: '{email}'")
    print(f"Spam Probability: {spam_probability:.4f}")
    if spam_probability > 0.5:
        print("Prediction: Spam\n")
    else:
        print("Prediction: Not Spam\n")
