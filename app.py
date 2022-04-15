import os
import requests

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
    app.logger.info(f"*** {r}")

    if r.status_code != 200:
        return jsonify({"models": []})

    r = r.json() 
    model_names = []
    for element in r["data"]:
        name = element["fine_tuned_model"]
        model_names.append(name)
    
    return jsonify({"models": model_names})


def generate_prompt(prompt):
    return f"{prompt} ->"
