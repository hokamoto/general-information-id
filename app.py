from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import numpy as np
import string
import sqlite3
import sqlite_vec
import json
from sqlite_vec import serialize_float32

# Initialize Flask app and SimCSE model
app = Flask(__name__)
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Base62 character set
BASE62_ALPHABET = string.digits + string.ascii_letters

# Helper functions
def quantize_to_n_bits(vector, n_bits):
    min_val, max_val = np.min(vector), np.max(vector)
    n_bins = 2 ** n_bits
    bins = np.linspace(min_val, max_val, n_bins)
    quantized_vector = np.digitize(vector, bins) - 1  # digitize starts from 1
    quantized_vector = np.clip(quantized_vector, 0, n_bins - 1)
    return quantized_vector

def base62_encode(num):
    # Encode integer to Base62 string
    if num == 0:
        return BASE62_ALPHABET[0]
    base62_str = ""
    while num > 0:
        num, remainder = divmod(num, 62)
        base62_str = BASE62_ALPHABET[remainder] + base62_str
    return base62_str

def quantized_vector_to_base62(quantized_vector, n_bits):
    binary_str = ''.join([format(qv, f'0{n_bits}b') for qv in quantized_vector])
    num = int(binary_str, 2)
    return base62_encode(num)

def base62_decode(base62_str):
    # Decode Base62 string to integer
    num = 0
    for char in base62_str:
        num = num * 62 + BASE62_ALPHABET.index(char)
    return num

def base62_to_quantized_vector(base62_str, n_bits, vector_length):
    # Base62 string to integer
    num = base62_decode(base62_str)
    # Convert integer to binary string
    binary_str = format(num, f'0{n_bits * vector_length}b')
    # Split binary string into quantized vector
    quantized_vector = [int(binary_str[i:i + n_bits], 2) for i in range(0, len(binary_str), n_bits)]
    return quantized_vector

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

# Route for computing average embedding and fingerprint
@app.route('/generate_fingerprint', methods=['POST'])
def generate_fingerprint():
    try:
        texts = request.form.get('texts', '').splitlines()
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400

        # Generate embeddings for each text
        embeddings = [model.encode(text) for text in texts]

        # Calculate the average vector
        average_vector = np.mean(embeddings, axis=0)

        # Quantize the average vector to 4 bits
        quantized_vector = quantize_to_n_bits(average_vector, 4)
        normalized_quantized_vector = quantized_vector / np.linalg.norm(quantized_vector)

        # Encode quantized vector to Base62 (fingerprint)
        fingerprint = quantized_vector_to_base62(quantized_vector, 4)

        # Return results as JSON
        return jsonify({
            'average_vector': normalized_quantized_vector.tolist(),
            'fingerprint': fingerprint,
            'shape': normalized_quantized_vector.shape
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    try:
        # Calculation the embedding from thq query
        query_text = request.form.get('query')
        if not query_text:
            return render_template('error.html', message="No query text provided")
        embedding = model.encode(query_text)
        quantized_vector = quantize_to_n_bits(embedding, 4)
        normalized_quantized_vector = quantized_vector / np.linalg.norm(quantized_vector)

        # Connect to the SQLite database
        conn = sqlite3.connect('giid.db')
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        cursor = conn.cursor()

        result = conn.execute('select giid, vec_distance_cosine(embedding, ?) as distance from vec_giid order by distance limit 1', [serialize_float32(normalized_quantized_vector)])
        res = result.fetchone()
        giid = res[0]
        distance = float(res[1])
        print(distance)
        if distance > 0.06:
            return render_template('error.html', message="Information not found")

        # Query for main giid entry data
        cursor.execute('''
            SELECT ge.date_identified, ge.narrative
            FROM giid_entries ge
            WHERE ge.giid = ?
        ''', (giid,))
        entry = cursor.fetchone()
        
        if entry is None:
            return render_template('error.html', message="No data found for the given GIID")

        # Load the narrative JSON array
        narrative_list = json.loads(entry[1])

        data = {
            'giid': giid,
            'date_identified': entry[0],
            'narrative': narrative_list,
            'language_distribution': [],
            'tags': [],
            'evaluations': []
        }

        # Query for language distribution
        cursor.execute('''
            SELECT language, percentage 
            FROM language_distribution
            WHERE giid = ?
        ''', (giid,))
        data['language_distribution'] = [{'language': row[0], 'percentage': row[1]} for row in cursor.fetchall()]

        # Query for tags
        cursor.execute('''
            SELECT DISTINCT tag
            FROM tags
            WHERE giid = ?
        ''', (giid,))
        data['tags'] = [row[0] for row in cursor.fetchall()]

        # Query for evaluations
        cursor.execute('''
            SELECT organization, evaluation, date_of_evaluation, comments, link
            FROM evaluations
            WHERE giid = ?
        ''', (giid,))
        data['evaluations'] = [
            {
                'organization': row[0],
                'evaluation': row[1],
                'date_of_evaluation': row[2],
                'comments': row[3],
                'link': row[4]
            }
            for row in cursor.fetchall()
        ]

        # Close the database connection
        conn.close()

        # Render the HTML template with the retrieved data
        return render_template('entry.html', data=data)

    except Exception as e:
        return render_template('error.html', message=str(e))

# Run the app
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=80)
