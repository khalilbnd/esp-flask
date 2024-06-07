from transformers import BloomTokenizerFast, BloomForCausalLM
import torch

# Load the tokenizer and model
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

def generate_question(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def parse_generated_text(generated_text):
    try:
        # Assuming the format is: "Question: ... Options: [opt1, opt2, opt3, opt4] Correct Answer Index: ..."
        parts = generated_text.split("Options:")
        if len(parts) < 2:
            raise ValueError("Generated text does not contain 'Options:'")

        question = parts[0].strip()
        options_part = parts[1].split("Correct Answer Index:")
        if len(options_part) < 2:
            raise ValueError("Generated text does not contain 'Correct Answer Index:'")

        options = [opt.strip() for opt in options_part[0].strip().split(",")]
        correct_answer_index = int(options_part[1].strip())

        return {
            "question": question,
            "options": options,
            "correctAnswerIndex": correct_answer_index
        }
    except Exception as e:
        print(f"Error parsing generated text: {generated_text}\nError: {e}")
        return None
def generate_and_print_question(prompt, max_length=100):
    generated_text = generate_question(prompt, max_length=max_length)
    question_data = parse_generated_text(generated_text)
    if question_data:
        print("Question:", question_data["question"])
        for idx, option in enumerate(question_data["options"], start=1):
            print(f"Option {idx}: {option}")
        print("Correct Answer Index:", question_data["correctAnswerIndex"])
    else:
        print("Failed to generate a valid question.")

# Define a prompt for question generation
prompt = "Generate a question about military law principles."

# Generate and print the question with options
generate_and_print_question(prompt)
