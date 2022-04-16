import os
import requests
import json

import openai
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/newmodel")
def new_model():
    return render_template("newmodel.html")


@app.route("/_run_model", methods=['POST'])
def run_model():
    prompt = request.form["prompt"]

    try:
        response = openai.Completion.create(
            model=request.form["model"],
            prompt=generate_prompt(prompt),
            temperature=0,
        )
    except openai.error.RateLimitError as e:
        app.logger.info(f"*** ERROR: {e}")
        return jsonify({"result": f"Error: {e}"})
    
    completion = response["choices"][0]["text"]

    return jsonify({"result": completion})


@app.route("/_models")
def models():
    url = "https://api.openai.com/v1/fine-tunes"
    headers = {'Authorization': f'Bearer {openai.api_key}'}

    r = requests.get(url, headers=headers)

    if r.status_code != 200:
        return jsonify({"models": []})

    r = r.json() 
    model_names = []
    for element in r["data"]:
        try:
            name = element["fine_tuned_model"]
        except Exception as e:
            app.logger.info(f"*** Model unavailable: {e}")
        else:
            model_names.append(name)
    
    return jsonify({"models": model_names})


@app.route("/_new_fine_tune", methods=['POST'])
def new_fine_tune():
    file_data = request.form["training-file"]
    try:
        ft_file = openai.File.create(file_data, "fine-tune")
    except Exception as e:
        return jsonify({"result": f"Failed to upload training file {e}."})
        
    file_id = ft_file["id"]
    base_model = request.form["model"]
    suffix = request.form["suffix"]

    data = { "training_file": file_id, "model": base_model, "suffix": suffix }

    url = "https://api.openai.com/v1/fine-tunes"
    headers = {
        'Authorization': f'Bearer {openai.api_key}',
        'Content-type': 'application/json',
    }
    
    r = requests.post(url, data=json.dumps(data), headers=headers)
    if r.status_code != 200:
        return jsonify({"result": f"Failed to create fine-tuned model {r}."})

    return jsonify({"result": "Model enqueued for training. It will be available soon (minutes)."})


def generate_prompt(prompt):
    return f"{prompt} ->"
