import joblib
import string
import nltk
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer



# Load the trained Extra Trees model
model = joblib.load('extra_trees_model.pkl')


# Load the cleaned text column from the pickle file
loaded_text = pd.read_pickle('cleaned_text.pkl')

# Load the saved TfidfVectorizer instance
#tf = joblib.load('tfidf_vectorizer.pkl')

ps = PorterStemmer()
# Function to preprocess the input text
def preprocess_text(text):
    # Apply the same text preprocessing steps as during training
    # Tokenization, lowercase conversion, removal of non-alphanumeric characters, etc.
    # Preprocess the text and return the processed text
    text = nltk.word_tokenize(text.lower())
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            stemming = ps.stem(i)
            y.append(stemming)

    return " ".join(y)



# Create the web interface using Streamlit
def main():
    # Set the title and description
    st.title('Spam Classifier')
    st.markdown('Enter a message to check if it is spam or not.')

    # Create an input field for the user to enter the message
    user_input = st.text_input('Enter a message')

    # Create a button to trigger the prediction
    if st.button('Predict'):
        # Preprocess the input message
        preprocessed_text = preprocess_text(user_input)

        # Transform the preprocessed text using the pre-trained TfidfVectorizer
        tf = TfidfVectorizer()
        # %%
        tf.fit(loaded_text)

        vectorized_text = tf.transform([preprocessed_text]).toarray()

        # Make the prediction using the trained model
        prediction = model.predict(vectorized_text)

        # Display the prediction result
        if prediction == 1:
            st.error('The message is classified as spam.')
        else:
            st.success('The message is not spam.')

# Run the web interface
if __name__ == '__main__':
    main()