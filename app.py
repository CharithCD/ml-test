from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load your models
svd_model = joblib.load('svd_model_02.pkl')
cosine_sim = joblib.load('cosine_similarity_truncated.pkl')
tfidf = joblib.load('tfidf_vectorizer_truncated.pkl')

# Load your dataset
df = pd.read_csv('balanced_interactions_02.csv')

# Initialize Flask app
app = Flask(__name__)

# Function to get collaborative recommendations
def get_collaborative_recommendations(user_id, model, df, top_n=5):
    all_items = df['article_id'].unique()
    predictions = [model.predict(user_id, item_id) for item_id in all_items]
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
    top_n_recommendations = [pred.iid for pred in top_predictions[:top_n]]
    return top_n_recommendations

# Function to get content-based recommendations
def get_content_based_recommendations(article_id, cosine_sim, df, top_n=5):
    idx = df.index[df['article_id'] == article_id]
    if len(idx) == 0:
        return []
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    similar_articles = [df.iloc[i[0]]['article_id'] for i in sim_scores]
    return similar_articles

# Combined recommendation function
def combine_recommendations(user_id, article_id, cosine_sim, df, cf_model, top_n=5):
    content_recs = get_content_based_recommendations(article_id, cosine_sim, df, top_n=top_n)
    cf_recs = get_collaborative_recommendations(user_id, cf_model, df, top_n=top_n)
    combined_recs = set(content_recs) | set(cf_recs)
    return list(combined_recs)[:top_n]

# Define a route for collaborative filtering recommendations
@app.route('/collaborative', methods=['GET'])
def collaborative():
    user_id = request.args.get('user_id')
    top_n = int(request.args.get('top_n', 5))
    recommendations = get_collaborative_recommendations(user_id, svd_model, df, top_n)
    return jsonify({'recommendations': recommendations})

# Define a route for content-based recommendations
@app.route('/content', methods=['GET'])
def content():
    article_id = request.args.get('article_id')
    top_n = int(request.args.get('top_n', 5))
    recommendations = get_content_based_recommendations(article_id, cosine_sim, df, top_n)
    return jsonify({'recommendations': recommendations})

# Define a route for combined recommendations
@app.route('/combined', methods=['GET'])
def combined():
    user_id = request.args.get('user_id')
    article_id = request.args.get('article_id')
    top_n = int(request.args.get('top_n', 5))
    recommendations = combine_recommendations(user_id, article_id, cosine_sim, df, svd_model, top_n)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
