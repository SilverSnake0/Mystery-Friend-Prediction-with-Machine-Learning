# Importing friends' writing samples:
from friend_doc import goldman_docs, henson_docs, wu_docs

# Importing necessary scikit-learn modules:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Combine friends' writing samples into one list:
friends_docs = goldman_docs + henson_docs + wu_docs

# Assign labels to the friends:
# - Goldman Emma = 1
# - Henson Matthew = 2
# - Wu Tingfang = 3
friends_labels = [1] * 154 + [2] * 141 + [3] * 166

# Print out a document from each friend:
print(f'Goldman Emma (Friend #1): "{goldman_docs[30]}"\n')
print(f'Henson Matthew (Friend #2): "{henson_docs[30]}"\n')
print(f'Wu Tingfang (Friend #3): "{wu_docs[30]}"\n')

# Set the document to classify:
mystery_postcard = """
My friend,
From the 10th of July to the 13th, a fierce storm raged, clouds of
freeing spray broke over the ship, incasing her in a coat of icy mail,
and the tempest forced all of the ice out of the lower end of the
channel and beyond as far as the eye could see, but the _Roosevelt_
still remained surrounded by ice.
Hope to see you soon.
"""

# Create a bag-of-words vectorizer:
bow_vectorizer = CountVectorizer()

# Vectorize the friends' writing samples:
friends_vectors = bow_vectorizer.fit_transform(friends_docs)

# Vectorize the mystery_postcard:
mystery_vector = bow_vectorizer.transform([mystery_postcard])

# Train a Naive Bayes classifier on the vectorized friends' writing samples:
friends_classifier = MultinomialNB()
friends_classifier.fit(friends_vectors, friends_labels)

# Predict the label of the mystery postcard and its probability for each label:
predictions = friends_classifier.predict(mystery_vector)
predictions_probability = friends_classifier.predict_proba(mystery_vector)

# Convert prediction probability to percentage:
predictions_probability_percentage = [
    f"{prob*100:.2f}%" for prob in predictions_probability[0]]

# Determine the friend who sent the postcard:
mystery_friend = predictions[0] if predictions[0] else "someone else"
friend_probability = mystery_friend - 1

# Print the results:
print(
    f"The postcard was from friend #{mystery_friend} with {predictions_probability_percentage[friend_probability]} probability!")
