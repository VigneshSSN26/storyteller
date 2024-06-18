from flask import Flask, request, render_template, jsonify
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
prompt_header=['horror','comedy','crime']
# Load the trained model and tokenizer
model = tf.keras.models.load_model("models/story_generator.h5")
with open("models/tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Function to generate a story from a given theme and word count
def generate_story(theme, word_count):
    seed_text = theme.lower()
    story = ""
    for _ in range(word_count):
        # Tokenize the seed text
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # Pad sequences
        token_list = pad_sequences([token_list], maxlen=model.input_shape[1], padding='pre')
        # Predict probabilities for each word
        predicted_probabilities = model.predict(token_list, verbose=0)[0]
        # Get the index of the word with the highest probability
        predicted_index = tf.argmax(predicted_probabilities, axis=-1).numpy()
        # Convert the index to the corresponding word
        output_word = tokenizer.index_word.get(predicted_index, "")
        # Update the seed text
        seed_text += " " + output_word
        story += output_word + " "
    return story.strip()

# Route to render the home page
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

# Route to generate a story based on the provided prompt
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt")
    if prompt:
        # Parse the prompt to extract theme and word count
        prompt_parts = prompt.split(" ")
        theme = ""
        word_count = 0
        for part in prompt_parts:
            if part.isdigit():
                word_count = int(part)
            else:
                if part in prompt_header:
                    theme=part
        # Generate story based on theme and word count
        story = generate_story(theme.strip(), word_count)
        return jsonify({"story": story})
    else:
        return jsonify({"error": "Please provide a prompt."})

if __name__ == "__main__":
    app.run(debug=True)


