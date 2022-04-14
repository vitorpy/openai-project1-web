import os
import requests

import openai
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/_run_model", methods=['POST'])
def run_model():
    prompt = request.form["prompt"]

    try:
        response = openai.Completion.create(
            model="davinci:ft-ouc-vitor-experimentation-2022-03-30-19-55-21",
            prompt=generate_prompt(prompt),
            temperature=0.6,
        )
    except openai.error.RateLimitError as e:
        app.logger.info(f"*** ERROR: {e}")
        return jsonify(f"Error: {e}")
    
    completion = response["choices"][0]["text"]

    return jsonify({"result": completion})


@app.route("/_models")
def models():
    url = "https://api.openai.com/v1/fine-tunes"
    headers = {'Authorization': f'Bearer {openai.api_key}'}

    r = requests.get(url, headers=headers)
    
    return jsonify(r)


def generate_prompt(prompt):
    return f"{prompt} ->"
