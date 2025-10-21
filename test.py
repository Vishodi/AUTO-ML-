from flask import Flask, request, jsonify, render_template
import pickle, uuid, os
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# In-memory storage for uploaded models (simple - ephemeral)
loaded_models = {}

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded (expected key "model")'}), 400
    file = request.files['model']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    try:
        obj = pickle.load(file)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to load pickle: {e}'}), 400

    # Expect packaged export object (as produced by app.py)
    if not isinstance(obj, dict) or 'model' not in obj or 'feature_columns' not in obj:
        return jsonify({'status': 'error',
                        'message': 'Pickle must be a packaged dict containing "model" and "feature_columns".'}), 400

    model = obj['model']
    feature_columns = obj['feature_columns']
    encoders = obj.get('feature_encoders', {}) or {}
    problem_type = obj.get('problem_type', 'Regression')

    # Extract encoder options for the UI (classes_ -> list)
    encoder_options = {}
    for col, le in encoders.items():
        try:
            encoder_options[col] = list(getattr(le, 'classes_', []))
        except Exception:
            encoder_options[col] = []

    model_id = str(uuid.uuid4())
    loaded_models[model_id] = {
        'model': model,
        'feature_columns': list(feature_columns),
        'encoders': encoders,
        'target_encoder': obj.get('target_encoder', None),
        'problem_type': problem_type
    }
    return jsonify({
        'status': 'ok',
        'model_id': model_id,
        'feature_columns': list(feature_columns),
        'encoder_options': encoder_options,
        'problem_type': problem_type
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'status': 'error', 'message': 'Expected JSON body'}), 400
    model_id = data.get('model_id')
    inputs = data.get('inputs', {})
    if not model_id or model_id not in loaded_models:
        return jsonify({'status': 'error', 'message': 'Model not found'}), 400

    model_info = loaded_models[model_id]
    cols = model_info['feature_columns']
    encoders = model_info['encoders']

    # Build a single-row DataFrame in the right column order
    row = []
    for col in cols:
        v = inputs.get(col, "")
        if v == "" or v is None:
            row.append(np.nan)
        else:
            if col in encoders:
                le = encoders[col]
                try:
                    # convert to string as encoders were trained on strings
                    transformed = le.transform([str(v)])[0]
                except Exception:
                    # unknown category: fallback to most frequent/first known class index
                    classes = getattr(le, 'classes_', [])
                    transformed = 0 if len(classes) == 0 else 0
                row.append(transformed)
            else:
                try:
                    row.append(float(v))
                except Exception:
                    # non-numeric - send NaN
                    row.append(np.nan)

    X = pd.DataFrame([row], columns=cols)

    try:
        model = model_info['model']
        pred = model.predict(X)
        result = None
        if model_info['problem_type'] == 'Classification' and model_info.get('target_encoder') is not None:
            try:
                # inverse transform if possible
                te = model_info['target_encoder']
                # prediction might be array-like of ints
                raw_val = pred[0]
                # ensure integer index for inverse_transform where needed
                try:
                    inv = te.inverse_transform([int(raw_val)])[0]
                except Exception:
                    inv = str(raw_val)
                result = {'prediction': inv, 'raw': raw_val}
            except Exception:
                result = {'prediction': str(pred[0]), 'raw': pred[0]}
        else:
            result = {'prediction': float(pred[0])}

        # include probabilities if classifier supports it
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(X).tolist()[0]
                result['probabilities'] = probs
            except Exception:
                pass

        return jsonify({'status': 'ok', **result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Prediction failed: {e}'}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 for container access; adjust debug as needed.
    app.run(host='0.0.0.0', port=5000, debug=True)
