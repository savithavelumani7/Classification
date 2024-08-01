# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:06:56 2024

@author: ELCOT
"""
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import cv2
import re
import pytesseract
from PIL import Image
import base64
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, accuracy_score,recall_score,classification_report
import seaborn as sns
import pickle
import warnings
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import norm
import plotly.express as px
from sklearn.feature_selection import SelectKBest, f_classif
from matplotlib.colors import ListedColormap
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from spacy import displacy
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

st.set_page_config(page_title="Machine Learning Model", layout="wide")

def back_img(image):
    with open(image, "rb") as image_file:
        encode_str = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encode_str.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

back_img("Back_img1.jpg")  

st.title('Machine Learning Model')


with st.sidebar:

    st.sidebar.image("icon.png",use_column_width=True)
    opt = option_menu("Title",
                      ["EDA", "Prediction","Evaluation Metrics","Image Processing","NLP Detailing","Customer Recomendation"],
                      menu_icon="cast",
                      styles={
                          "container": {"padding":"4!important", "background-color":"lightblue"},
                          "icon":{"color":"#01A982","font-size":"20px"},
                          "nav-link": {"font-size": "20px", "text-align":"left"},
                          "nav-link-selected": {"background-color": "blue"}
                      }
                      )
    

if opt == "EDA":
    # Step 1: Load CSV File
        class SessionState:
            def __init__(self, **kwargs):
                for key, val in kwargs.items():
                    setattr(self, key, val)

# Create an instance of SessionState
        session_state = SessionState(df=None)
        
        # Step 1: Load CSV File
        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            session_state.df = df  # Save the data in the session state
        
        # Step 2: Display DataFrame
        if session_state.df is not None and st.button("Show DataFrame"):
            st.dataframe(session_state.df)
      
        if session_state.df is not None:
            st.write("### DataFrame")
            st.dataframe(session_state.df)
        
        # Drop Duplicates and NaN Values
        if session_state.df is not None:
            st.write("### Drop Duplicates and NaN Values")
            session_state.df = session_state.df.drop_duplicates()
            session_state.df = session_state.df.dropna()
            st.dataframe(session_state.df)
            st.success("Duplicates and NaN values dropped successfully!")
        
        
            
        if session_state.df is not None:
            st.write("### Summary Statistics")
            st.text(session_state.df.describe())

        
        # Label Encoding
        if session_state.df is not None:
            st.write("### Label Encoding")
            le = LabelEncoder()
            for col in session_state.df.columns:
                if session_state.df[col].dtype == 'object' or session_state.df[col].dtype == 'bool':
                    session_state.df[col] = le.fit_transform(session_state.df[col])
            st.dataframe(session_state.df)
            st.success("Label Encoding completed successfully!")
        
        # One-Hot Encoding for categorical columns
        if session_state.df is not None:
            st.write("### One-Hot Encoding")
            categorical_columns = session_state.df.select_dtypes(include=['object']).columns
            session_state.df = pd.get_dummies(session_state.df, columns=categorical_columns)
            st.dataframe(session_state.df)
            st.success("One-Hot Encoding completed successfully!")
        
        # DateTime Format Conversion
        if session_state.df is not None:
            st.write("### DateTime Format Conversion")
            session_state.df['target_date'] = pd.to_datetime(session_state.df['target_date'])
            st.dataframe(session_state.df)
            st.success("DateTime Format Conversion completed successfully!")
        
        # Plot Relationship Curve
        if session_state.df is not None:
            st.write("### Plot Relationship Curve")
            sampled_df = pd.DataFrame(session_state.df["avg_visit_time"].sample(min(1000, len(session_state.df))))
            sns.pairplot(sampled_df)
            st.pyplot()
            # Disable the warning about PyplotGlobalUse
            st.set_option('deprecation.showPyplotGlobalUse', False)
        
        # Detect and Treat Outliers
        if session_state.df is not None:
            st.write("### Detect and Treat Outliers")
            Q1 = session_state.df['transactionRevenue'].quantile(0.25)
            Q3 = session_state.df['transactionRevenue'].quantile(0.75)
            IQR = Q3 - Q1
            session_state.df = session_state.df[~((session_state.df['transactionRevenue'] < (Q1 - 1.5 * IQR)) | (session_state.df['transactionRevenue'] > (Q3 + 1.5 * IQR)))]
            st.dataframe(session_state.df)
            st.success("Outliers detected and treated successfully!")
        
        # Plot Normalization Curve
        if session_state.df is not None:
            st.write("### Plot Normalization Curve")
            sns.histplot(session_state.df['avg_session_time'], kde=True)
            st.pyplot()
            # Disable the warning about PyplotGlobalUse
            st.set_option('deprecation.showPyplotGlobalUse', False)
        
        # Treat Skewness
        if session_state.df is not None:
            st.write("### Treat Skewness")
            session_state.df['latest_visit_number'] = np.log1p(session_state.df['latest_visit_number'])
            st.dataframe(session_state.df)
            st.success("Skewness treated successfully!")
        
        # Calculate Correlation and Plot Heatmap
        if session_state.df is not None:
            st.write("### Correlation Heatmap")
            correlation_matrix = session_state.df.corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
            st.pyplot()
            # Disable the warning about PyplotGlobalUse
            st.set_option('deprecation.showPyplotGlobalUse', False)
                
        
        # Box Plot for Outlier Detection
        if session_state.df is not None:
            st.write("### Box Plot for Outlier Detection")
            numeric_columns = session_state.df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                sns.boxplot(x=col, data=session_state.df)
                st.pyplot()
            # Disable the warning about PyplotGlobalUse
            st.set_option('deprecation.showPyplotGlobalUse', False)
        
      

if opt=="Prediction":
    c1,c2,c3,c4 = st.columns(4)
    st.markdown("""
    <style>
        .st-ax {
                background-color: lightblue;
        }

        .stTextInput input{
                background-color: lightblue;
        }
                
        .stNumberInput input{
                background-color: lightblue;
        }
        .stDateInput input{
                background-color: lightblue;
        }
                
    </style>
    """,unsafe_allow_html=True)
    with open("model_rf.pkl", "rb") as mf:
        new_model = pickle.load(mf)

    # Input form
    with st.form("user_inputs"):
        with st.container():
            count_session = st.number_input("count_session")
            time_earliest_visit = st.number_input("time_earliest_visit")
            avg_visit_time = st.number_input("avg_visit_time")
            days_since_last_visit = st.number_input("days_since_last_visit")
            days_since_first_visit = st.number_input("days_since_first_visit")
            visits_per_day = st.number_input("visits_per_day")
            bounce_rate = st.number_input("bounce_rate")
            earliest_source = st.number_input("earliest_source")
            latest_source = st.number_input("latest_source")
            earliest_medium = st.number_input("earliest_medium")
            latest_medium = st.number_input("latest_medium")
            earliest_keyword = st.number_input("earliest_keyword")
            latest_keyword = st.number_input("latest_keyword")
            earliest_isTrueDirect = st.number_input("earliest_isTrueDirect")
            latest_isTrueDirect = st.number_input("latest_isTrueDirect")
            num_interactions = st.number_input("num_interactions")
            bounces = st.number_input("bounces")
            time_on_site = st.number_input("time_on_site")
            time_latest_visit = st.number_input("time_latest_visit")
    
        submit_button = st.form_submit_button(label="Submit")
    
    # Predict using the model
    if submit_button:
        test_data = np.array([
            [
                count_session, time_earliest_visit, avg_visit_time, days_since_last_visit, 
                days_since_first_visit, visits_per_day, bounce_rate, earliest_source, 
                latest_source, earliest_medium, latest_medium, earliest_keyword, 
                latest_keyword, earliest_isTrueDirect, latest_isTrueDirect, num_interactions, 
                bounces, time_on_site, time_latest_visit
            ]
        ])
        
        # Convert the data to float
        test_data = test_data.astype(float)
    
        # Make predictions
        predicted = new_model.predict(test_data)[0]
        prediction_proba = new_model.predict_proba(test_data)
    
        # Display the results
        st.write("Prediction:", predicted)
        st.write("Prediction Probability:", prediction_proba)

if opt=="Evaluation Metrics":

    # Step 1: Load CSV File
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    
        # EDA and Preprocessing Steps
    
        # Duplicate Removal
        st.write("### Duplicate Removal")
        df1 = df.drop_duplicates()
        st.success("Duplicates removed successfully!")
    
        # NaN Value Fill
        st.write("### NaN Value Fill")
        df2 = df1.fillna(0)  # You can replace 0 with the desired value
        st.success("NaN values filled successfully!")
    
        # DateTime Format Conversion
        st.write("### DateTime Format Conversion")
        date_columns = df2.select_dtypes(include=['datetime']).columns
        for col in date_columns:
            df2[col] = pd.to_datetime(df2[col])
        st.success("DateTime Format Conversion completed successfully!")
    
        # Display DataFrame
        st.dataframe(df2)
    
        # Summary Statistics
        st.write("### Summary Statistics")
        st.write(df2.describe())
    
        # Feature Importance with Random Forest
        le = LabelEncoder()
        for col in df2.columns:
            if df2[col].dtype == 'object' or df2[col].dtype == 'bool':
                df2[col] = le.fit_transform(df2[col])
    
        X_train = df2.drop('has_converted', axis=1)
        y_train = df2['has_converted']
    
        # Plot feature importance
        st.write("### Feature Importance with Random Forest")
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        feature_importances = rf.feature_importances_
        
        feature_importance_df=pd.DataFrame({
            "Feature":X_train.columns,
            "Impotance":feature_importances
            })
        top_10_features=feature_importance_df.sort_values(by="Impotance",ascending=False).head(10)["Feature"].tolist()
        extra_feature="has_converted"
        df3 = df2[top_10_features + [extra_feature]]
        #columns=['count_session','time_earliest_visit','avg_visit_time','days_since_last_visit','days_since_first_visit','visits_per_day','bounce_rate','earliest_source','latest_source','earliest_medium','has_converted']
        
        # Streamlit code
        st.title('Top 10 Features Importance')
        st.bar_chart(top_10_features)
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
    
        # Pie Chart using Feature Importance
        st.write("### Pie Chart using Feature Importance")
        fig, ax = plt.subplots()
        ax.pie(feature_importances, labels=feature_importances, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
        
        # Random Forest Model Build
        #df3 = pd.DataFrame(df3)

        # Drop the 'has_converted' column
        X = df3.drop('has_converted', axis=1)
        y=df3['has_converted']
        
            
        # Random Forest Model Build
        model = RandomForestClassifier(n_estimators=50,random_state=42)
        
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        rf_predict=rf_model.predict(X_train)
        rf_accuracy = accuracy_score(y_train,rf_predict)
        rf_Precision=precision_score(y_train,rf_predict)
        rf_recall=recall_score(y_train,rf_predict)
        rf_f1=f1_score(y_train,rf_predict)
        
        # Display Random Forest Model results
        st.write("# Random Forest Model")
        st.write("Accuracy:", rf_accuracy)
        st.write("Precision:", rf_Precision)
        st.write("Recall:", rf_recall)
        st.write("F1_score:", rf_f1)
        
        # Decision Tree Model Build
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)
        dt_predict=dt_model.predict(X_train)
        dt_accuracy = accuracy_score(y_train,dt_predict)
        dt_Precision=precision_score(y_train,dt_predict)
        dt_recall=recall_score(y_train,dt_predict)
        dt_f1=f1_score(y_train,dt_predict)
        
        # Display Decision Tree Model results
        st.write("# Decision Tree Model")
        st.write("Accuracy:", dt_accuracy)
        st.write("Precision:", dt_Precision)
        st.write("Recall:", dt_recall)
        st.write("F1_score:", dt_f1)

    
        # KNN Model Build
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)  # Use the X_train, y_train from the first block
        knn_predict=knn_model.predict(X_train)
        knn_accuracy = accuracy_score(y_train, knn_predict)
        knn_Precision=precision_score(y_train,knn_predict)
        knn_recall=recall_score(y_train,knn_predict)
        knn_f1=f1_score(y_train,knn_predict)
        
        
        # Display KNN Model results
        st.write("# KNN Model")
        st.write("Accuracy:", knn_accuracy)
        st.write("Precision:",knn_Precision)
        st.write("Recall:", knn_recall)
        st.write("F1_score:",knn_f1)
    
    # Display results in a table
        results_data = {
            'Model': ['Random Forest', 'Decision Tree', 'KNN'],
            'Accuracy': [rf_accuracy, dt_accuracy, knn_accuracy],
            'Precision': [rf_Precision, dt_Precision, knn_Precision],
            'Recall': [rf_recall, dt_recall, knn_recall],
            'F1_score': [rf_f1, dt_f1, knn_f1]
        }
        
        results_table = st.table(results_data)
            
    
        # Plotly Visualization
        fig = px.bar(
            x=['Random Forest', 'Decision Tree', 'KNN'],
            y=[rf_accuracy, dt_accuracy, knn_accuracy],
            labels={'y': 'Accuracy', 'x': 'Models'},
            title='Model Accuracy Comparison'
        )
    
        st.plotly_chart(fig)


if opt == "Image Processing":
   


    # Function to preprocess image before OCR
    def preprocess_image(image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Apply thresholding to enhance text
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        
        text = pytesseract.image_to_string(thresh)
        
        # Display extracted text
        st.write("Extracted Text:")
        st.write(text)

        return thresh
    
    # Function to convert image to grayscale
    def convert_to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Function to resize image
    def resize_image(image, width):
        # Calculate the aspect ratio
        aspect_ratio = image.shape[1] / image.shape[0]
    
        # Calculate the new height based on the aspect ratio
        height = int(width / aspect_ratio)
    
        # Resize the image
        resized_image = cv2.resize(image, (width, height))
    
        return resized_image
    
    # Function to rotate image
    def rotate_image(image, angle):
        pil_image = Image.fromarray(image)
        rotated_image = pil_image.rotate(angle)
        return np.array(rotated_image)
    
    # Function to crop image
    def crop_image(image, x_start, y_start, x_end, y_end):
        cropped_image = image[y_start:y_end, x_start:x_end]
        return cropped_image
    
    # Function to create a mirror image
    def mirror_image(image):
        mirrored_image = cv2.flip(image, 1)  # 1 indicates horizontal flip
        return mirrored_image
    
    # Function to adjust image brightness
    def adjust_brightness(image, factor):
        # Convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
        # Multiply the V (Value) channel by the brightness factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
    
        # Convert the image back to BGR color space
        brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
        return brightened_image
    
    # Function for edge detection using Canny
    def edge_detection(image, low_threshold, high_threshold):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
        # Apply Canny edge detector
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
        return edges
    
    # Function for image sharpening using Laplacian kernel
    
    
    # Function for applying a mask to the image
    def apply_mask(image, mask):
        # Convert the mask to a 3-channel image for compatibility
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(image, mask)
    
        return masked_image
    
    # Your Streamlit app
    def main():
        st.title("Streamlit App with Image Processing Steps")
    
        # Example: File uploader
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
        if uploaded_file is not None:
            # Read the image
            image = cv2.imread(uploaded_file.name)
    
            # Image processing options
            options = st.multiselect(
                "Choose Image Processing Steps",
                ["Preprocess", "Grayscale", "Resize", "Rotate", "Crop", "Mirror", "Brightness", "Edge Detection", "Sharpen", "Mask"],
                default=["Preprocess"]
            )
    
            # Process the image based on selected options
            for option in options:
                if option == "Preprocess":
                    image = preprocess_image(image)
                elif option == "Grayscale":
                    image = convert_to_grayscale(image)
                elif option == "Resize":
                    desired_width = st.slider("Select Width for Resize", min_value=50, max_value=800, value=400, step=50)
                    image = resize_image(image, desired_width)
                elif option == "Rotate":
                    desired_angle = st.slider("Select Angle for Rotation", min_value=0, max_value=360, value=45, step=15)
                    image = rotate_image(image, desired_angle)
                elif option == "Crop":
                    x_start = st.number_input("X Start:", min_value=0, max_value=image.shape[1], value=0)
                    y_start = st.number_input("Y Start:", min_value=0, max_value=image.shape[0], value=0)
                    x_end = st.number_input("X End:", min_value=0, max_value=image.shape[1], value=image.shape[1])
                    y_end = st.number_input("Y End:", min_value=0, max_value=image.shape[0], value=image.shape[0])
                    image = crop_image(image, int(x_start), int(y_start), int(x_end), int(y_end))
                elif option == "Mirror":
                    image = mirror_image(image)
                elif option == "Brightness":
                    brightness_factor = st.slider("Adjust Brightness", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
                    image = adjust_brightness(image, brightness_factor)
                elif option == "Edge Detection":
                    low_threshold = st.slider("Low Threshold", min_value=0, max_value=255, value=50)
                    high_threshold = st.slider("High Threshold", min_value=0, max_value=255, value=150)
                    image = edge_detection(image, low_threshold, high_threshold)
                
                elif option == "Mask":
                    mask_file = st.file_uploader("Choose a mask file (binary image)", type=["png"])
                    if mask_file is not None:
                        mask = cv2.imread(mask_file.name, cv2.IMREAD_GRAYSCALE)
                        image = apply_mask(image, mask)
    
            # Display the processed image
            st.image(image, caption="Processed Image", use_column_width=True)
    
    if __name__ == "__main__":
        main()
    

if opt =="NLP Detailing":
    


    # Download NLTK resources
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # Download spaCy model
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
    # Sample data for text classification
    # Your text data
    text_data = [
        ("I love streamlit", "positive"),
        ("Streamlit is easy to use", "positive"),
        ("NLP processing in streamlit is great", "positive"),
        ("Streamlit helps in building interactive web apps", "positive"),
        ("I dislike bugs in streamlit", "negative"),
        ("Streamlit could improve in some areas", "negative"),
        ("NLP can be challenging for beginners", "negative"),
        ("I struggle with streamlit syntax", "negative"),
    ]
    
    df = pd.DataFrame(text_data, columns=['text', 'label'])
    
    X_train=df["text"]
    y_train=df["label"]
    
    # Streamlit app
    st.title("NLP Processing with Streamlit")
    
    # Text input
    text_input = st.text_area("Enter text for NLP processing:")
    
    # Tokenization
    if st.checkbox("Tokenization"):
        tokens = word_tokenize(text_input)
        st.write("Tokens:", tokens)
    
    # Stopword Removal
    if st.checkbox("Stopword Removal"):
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        st.write("Tokens after stopword removal:", filtered_tokens)
    
    # Number Removal
    if st.checkbox("Number Removal"):
        filtered_tokens = [word for word in filtered_tokens if not word.isdigit()]
        st.write("Tokens after number removal:", filtered_tokens)
    
    # Special Character Removal
    if st.checkbox("Special Character Removal"):
        filtered_tokens = [word for word in filtered_tokens if word.isalnum()]
        st.write("Tokens after special character removal:", filtered_tokens)
    
    # Stemming
    if st.checkbox("Stemming"):
        porter_stemmer = PorterStemmer()
        stemmed_tokens = [porter_stemmer.stem(word) for word in filtered_tokens]
        st.write("Tokens after stemming:", stemmed_tokens)
    
    # Lemmatization
    if st.checkbox("Lemmatization"):
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        st.write("Tokens after lemmatization:", lemmatized_tokens)
    
    # Parts of Speech (POS)
    if st.checkbox("Parts of Speech (POS)"):
        doc = nlp(text_input)
        pos_tags = [(token.text, token.pos_) for token in doc]
        st.write("Parts of Speech:", pos_tags)
    
   
    # Text Classification
    if st.checkbox("Text Classification"):
    # Create a pipeline with CountVectorizer and MultinomialNB
       
        model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])
    
        # Fit the model on the training data
        model.fit(X_train, y_train)
    
        # Make predictions on the testing data
        y_pred = model.predict(X_train)
    
        # Display evaluation metrics
        accuracy = accuracy_score(y_train, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Classification Report:\n", classification_report(y_train, y_pred))
    
    # Sentiment Analysis
    if st.checkbox("Sentiment Analysis"):
        #Assuming binary sentiment classification (positive and negative)
        sentiment = "Positive" if model.predict([text_input])[0] == "positive" else "Negative"
        st.write(f"Sentiment: {sentiment}")
    
    # Word Cloud
    if st.checkbox("Word Cloud"):
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_input)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()
    
    # Keyword Extraction
    if st.checkbox("Keyword Extraction"):
        keywords = nlp(text_input).ents
        st.write("Keywords:", [keyword.text for keyword in keywords])
    
    # Named Entity Recognition (NER)
    if st.checkbox("Named Entity Recognition (NER)"):
        doc_ner = nlp(text_input)
        ner_displacy = displacy.render(doc_ner, style="ent", page=True)
        st.write(ner_displacy, unsafe_allow_html=True)

    
    
if opt =="Customer Recomendation":
    


    # Load CSV data
    def load_data():
        # Replace 'your_data.csv' with the actual path to your CSV file
        df = pd.read_csv('market_data1.csv')
        return df
    
    # Create Surprise dataset
    def create_surprise_dataset(data):
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(data[['CustomeId','Products','Rating']], reader)
        return dataset
    
    # Build collaborative filtering model
    def build_collaborative_filtering_model(dataset):
        trainset, testset = train_test_split(dataset, test_size=0.2)
        model = SVD()
        model.fit(trainset)
        return model, testset
    
    # Main function
    def main():
        st.title('Customer Recommendation App with Surprise')
    
        # Load CSV data
        data = load_data()
    
        # Display original data
        st.write('### Original Data')
        st.write(data)
    
        # Create Surprise dataset
        dataset = create_surprise_dataset(data)
    
        # Build collaborative filtering model
        model, testset = build_collaborative_filtering_model(dataset)
    
        # Select a customer for recommendations
        selected_customer = st.selectbox('Select a CustomerId for recommendations:', data['CustomeId'].unique())
    
        # Generate recommendations
        if st.button('Generate Recommendations'):
            # Get unrated items for the selected customer
            unrated_items = data[data['CustomeId'] == selected_customer][['Products']]
            unrated_items = list(unrated_items['Products'])
    
            # Make predictions for unrated items
            predictions = [model.predict(selected_customer, item) for item in unrated_items]
    
            # Display top recommendations
            st.subheader(f'Top Recommendations for CustomerId {selected_customer}:')
            for prediction in sorted(predictions, key=lambda x: x.est, reverse=True)[:5]:
                st.write(f'Recommendations: {prediction.iid}')
    
    if __name__ == '__main__':
        main()
        