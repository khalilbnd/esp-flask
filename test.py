import spacy
import numpy as np
from gensim.models import KeyedVectors

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load pre-trained Word2Vec model
word_vectors = KeyedVectors.load_word2vec_format("word2vec.bin", binary=True)

def calculate_similarity(ref_text, input_text):
    input_doc = nlp(input_text)
    input_embeddings = [word_vectors[token.text] for token in input_doc if token.text in word_vectors]

    ref_doc = nlp(ref_text)
    ref_embeddings = [word_vectors[token.text] for token in ref_doc if token.text in word_vectors]

    if input_embeddings and ref_embeddings:
        similarity_score = np.dot(np.mean(input_embeddings, axis=0), np.mean(ref_embeddings, axis=0)) / \
                            (np.linalg.norm(np.mean(input_embeddings, axis=0)) * np.linalg.norm(np.mean(ref_embeddings, axis=0)))
        return similarity_score
    else:
        return 0.0

def generate_questions(field, subfield, level):
    # Define question templates for each category and level
    question_templates = {
        ('Computer Science', 'Artificial Intelligence', 'Basic'): [
            "What is the definition of [concept]?",
            "Explain the concept of [concept].",
            "What are the applications of [concept]?"
        ],
        ('Computer Science', 'Artificial Intelligence', 'Advanced'): [
            "Discuss the differences between [concept1] and [concept2].",
            "How does [concept] relate to [subfield]?",
            "Explain the algorithm behind [concept]."
        ],
        # Add more templates for other categories, subfields, and levels
    }

    # Select question templates based on field, subfield, and level
    selected_templates = question_templates.get((field, subfield, level), [])

    # Generate questions by replacing placeholders in templates with specific concepts
    questions = []
    for template in selected_templates:
        # Replace [concept] placeholder with specific concepts (you can define these based on your knowledge base)
        # For simplicity, let's use some example concepts here
        concept = "example_concept"
        question = template.replace("[concept]", concept).replace("[concept1]", "example_concept1").replace("[concept2]", "example_concept2").replace("[subfield]", subfield)
        questions.append(question)

    return questions

# Example usage
field = "Computer Science"
subfield = "Artificial Intelligence"
level = "Basic"

questions = generate_questions(field, subfield, level)
for i, question in enumerate(questions, start=1):
    print(f"Question {i}: {question}")
