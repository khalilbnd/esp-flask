from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import spacy
from nltk.corpus import wordnet

# Load models
model = SentenceTransformer('albert-base-v2')
nlp = spacy.load('en_core_web_sm')

# Initialize Flask app
app = Flask(__name__)

# def calculate_similarity(input_embedding, ref_embeddings):
#     cosine_scores = util.cos_sim(input_embedding, ref_embeddings)
#     max_similarity = cosine_scores.max().item()
#     return max_similarity

# def extract_key_concepts(text):
#     doc = nlp(text)
#     entities = {ent.text.lower() for ent in doc.ents}
#     noun_chunks = {chunk.text.lower() for chunk in doc.noun_chunks}
#     verbs = {token.lemma_.lower() for token in doc if token.pos_ in {"VERB", "ADJ", "ADV"}}
#     return entities.union(noun_chunks).union(verbs)

# def get_synonyms(word):
#     synonyms = set()
#     for syn in wordnet.synsets(word):
#         for lemma in syn.lemmas():
#             synonyms.add(lemma.name().lower().replace('_', ' '))
#     return synonyms

# def expand_with_synonyms(concepts):
#     expanded_concepts = set()
#     for concept in concepts:
#         expanded_concepts.add(concept)
#         expanded_concepts.update(get_synonyms(concept))
#     return expanded_concepts

def calculate_accuracy(input_text, ref_texts):
    # Calculate semantic similarity
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    ref_embeddings = model.encode(ref_texts, convert_to_tensor=True)
    
    
  

    # Get the maximum similarity for each input concept across all reference concepts
    cosine_scores = util.cos_sim(input_embedding, ref_embeddings)
    
    avg_concept_similarity = cosine_scores.item()
    
    
    return round(avg_concept_similarity, 2)

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    ref_texts = data.get('ref_texts')
    input_text = data.get('input_text')
    
    if not ref_texts or not input_text:
        return jsonify({'error': 'Both reference texts and input text are required'}), 400
    
    accuracy = calculate_accuracy(input_text, ref_texts)
    
    return jsonify({
        'ref_texts': ref_texts,
        'input_text': input_text,
        'accuracy': accuracy
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000)