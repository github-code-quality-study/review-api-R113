import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
ALLOWED_LOCATIONS = {
                "Albuquerque, New Mexico",
                "Carlsbad, California",
                "Chula Vista, California",
                "Colorado Springs, Colorado",
                "Denver, Colorado",
                "El Cajon, California",
                "El Paso, Texas",
                "Escondido, California",
                "Fresno, California",
                "La Mesa, California",
                "Las Vegas, Nevada",
                "Los Angeles, California",
                "Oceanside, California",
                "Phoenix, Arizona",
                "Sacramento, California",
                "Salt Lake City, Utah",
                "Salt Lake City, Utah",
                "San Diego, California",
                "Tucson, Arizona"
            }

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        try:
            self.reviews = pd.read_csv('data/reviews.csv').to_dict('records')
            logging.info("Reviews loaded successfully.")
        except Exception as ex:
            logging.error(f"Error loading reviews: {ex}")
            self.reviews = []

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        try:

            if environ["REQUEST_METHOD"] == "GET":

                query = parse_qs(environ["QUERY_STRING"])
                location = query.get("location", [None])[0]
                start_date = query.get("start_date", [None])[0]
                end_date = query.get("end_date", [None])[0]

                filtered_reviews = self.reviews

                if location:
                    if location not in ALLOWED_LOCATIONS:
                        start_response("400 Bad Request", [("Content-Type", "application/json")])
                        return [json.dumps({"error": f"'{location}' is not an allowed location."}).encode("utf-8")]
                    filtered_reviews = [review for review in filtered_reviews if review["Location"] == location]

                if start_date:
                    start_date = datetime.strptime(start_date, '%Y-%m-%d')
                    filtered_reviews = [
                        review for review in filtered_reviews
                        if datetime.strptime(review['Timestamp'], TIMESTAMP_FORMAT) >= start_date
                    ]

                if end_date:
                    end_date = datetime.strptime(end_date, '%Y-%m-%d')
                    filtered_reviews = [
                        review for review in filtered_reviews
                        if datetime.strptime(review['Timestamp'], TIMESTAMP_FORMAT) <= end_date
                    ]
            
                updated_reviews = []
                for review in filtered_reviews:
                    sentiment = self.analyze_sentiment(review["ReviewBody"])
                    updated_reviews.append({
                        "ReviewId": review["ReviewId"],
                        "Location": review["Location"],
                        "Timestamp": review["Timestamp"],
                        "ReviewBody": review["ReviewBody"],
                        "sentiment": sentiment
                    })
                
                reviews = sorted(updated_reviews, key=lambda x: x["sentiment"]["compound"], reverse=True)

                response_body = json.dumps(reviews, indent=2).encode("utf-8")
                

                start_response("200 OK", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                
                return [response_body]


            elif environ["REQUEST_METHOD"] == "POST":

                try:
                    content_length = int(environ.get('CONTENT_LENGTH', 0))
                    body = environ['wsgi.input'].read(content_length).decode("utf-8")
                    data = parse_qs(body)
                    
                    location = data.get("Location", [None])[0]
                    review_body = data.get("ReviewBody", [None])[0]

                    if not location or not review_body:
                        start_response("400 Bad Request", [("Content-Type", "application/json")])
                        return [json.dumps({"error": "Both 'Location' and 'ReviewBody' are required."}).encode("utf-8")]
                    
                    if location not in ALLOWED_LOCATIONS:
                        start_response("400 Bad Request", [("Content-Type", "application/json")])
                        return [json.dumps({"error": f"'{location}' is not an allowed location."}).encode("utf-8")]

                    new_review = {
                        "ReviewId": str(uuid.uuid4()),
                        "ReviewBody": review_body,
                        "Location": location,
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    self.reviews.append(new_review)

                    response_body = json.dumps(new_review, indent=2).encode("utf-8")
                    start_response("201 Created", [("Content-Type", "application/json")])
                    return [response_body]
                
                except json.JSONDecodeError:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Invalid JSON format."}).encode("utf-8")]
                            
                except Exception as ex:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": str(ex)}).encode("utf-8")]
            else:
                start_response("405 Method Not Allowed", [("Content-Type", "application/json")])
                return [json.dumps({"error": "Method not allowed."}).encode("utf-8")]
            
        except Exception as ex:
            logging.error(f"Error handling request: {ex}")
            start_response('500 Internal Server Error', [('Content-Type', 'application/json')])
            return [json.dumps({"error": "Internal server error."}).encode('utf-8')]


if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()