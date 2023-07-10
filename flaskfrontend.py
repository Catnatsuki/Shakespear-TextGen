from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
import pathlib
import os

app = Flask(__name__, static_folder='static')

cache_dir = './tmep'
dataset_file_name = 'shakespeare.txt'
dataset_file_origin = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

dataset_file_path = tf.keras.utils.get_file(
    fname=dataset_file_name,
    origin=dataset_file_origin,
    cache_dir=pathlib.Path(cache_dir).absolute()
)

print(dataset_file_path)


# Reading the database file.
text = open(dataset_file_path, mode='r').read()

print('Length of text: {} characters'.format(len(text)))

# The unique characters in the file
vocab = sorted(set(text))

print('{} unique characters'.format(len(vocab)))
print('vocab:', vocab)

# Map characters to their indices in vocabulary.
char2index = {char: index for index, char in enumerate(vocab)}

# Map character indices to characters from vacabulary.
index2char = np.array(vocab)


# Convert chars in text to indices.
text_as_int = np.array([char2index[char] for char in text])


# Length of the vocabulary in chars.
vocab_size = len(vocab)

# The embedding dimension.
embedding_dim = 256

# Number of RNN units.
rnn_units = 1024

model_path=os.path.join(sys.path[0], "tex_gen_final.h5")
model = load_model(model_path)

def generate_text(model, start_string, num_generate = 1000, temperature=1.0):
    # Evaluation step (generating text using the learned model)
    

    # Converting our start string to numbers (vectorizing).
    input_indices = [char2index[s] for s in start_string]
    input_indices = tf.expand_dims(input_indices, 0)

    # Empty string to store our results.
    text_generated = []

    # Here batch size == 1.
    model.reset_states()
    for char_index in range(num_generate):
        predictions = model(input_indices)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Using a categorical distribution to predict the character returned by the model.
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
        predictions,
        num_samples=1
        )[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state.
        input_indices = tf.expand_dims([predicted_id], 0)

        text_generated.append(index2char[predicted_id])

    return (start_string + ''.join(text_generated))

# define a function to handle incoming HTTP requests
def handle_request(request):
    if request.method == 'POST':
        form_data = request.form
        start_string = form_data.get('start_string')
        num_generate = int(form_data.get('num_generate'))
        temperature = float(form_data.get('temperature'))
        result = generate_text(model, start_string, num_generate, temperature)
        return render_template('result.html', result=result)
    else:
        return render_template('form.html')




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    start_string = str(request.form['start_string'])
    num_generate = int(request.form['num_generate'])
    temperature = float(request.form['temperature'])

    generated_text = generate_text(model, start_string, num_generate=num_generate, temperature=temperature)

    # add HTML tags for formatting
    generated_text = generated_text.replace('\n', '<br>')

    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
