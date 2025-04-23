
# Mikro's LSTM Chatbot

Mikro is a deep learning-powered chatbot built using Keras and LSTM (Long Short-Term Memory) networks. It's designed to understand and respond to natural language queries related to topics like Airbus, human anatomy, and more!



## Screenshot

[![LSTM_mikro_UI.png](https://i.postimg.cc/dVR4B1Xd/Screenshot-2025-04-22-181835.png)](https://postimg.cc/7563PxqY)


## Project Structure

To run tests, run the following command

```bash
─ mikro.ipynb               # Main Jupyter Notebook for model training and chatbot logic
─ Mikro_LSTM_Dataset.json   # Intents dataset with tags, patterns, and responses
─ config.toml               # Streamlit UI theme configuration
─ README.md                 # Project documentation (this file)
```


## Model Overview

- Model Type: LSTM Neural Network

- Framework: TensorFlow/Keras

- Functionality: Classifies user input into intent categories and responds appropriately



## Dataset
The chatbot uses a JSON-based dataset with a structure like:

```json
{
  "tag": "greeting",
  "patterns": ["Hello", "Hi", "Hey"],
  "responses": ["Hello! Welcome to the chatbot."]
}
```


## How to Run

1. Clone the repo:

```bash
  git clone https://github.com/yourusername/mikro-chatbot.git
cd mikro-chatbot

```
2. Install dependencies:
```bash
  pip install -r requirements.txt

```
3. Launch the notebook and run all cells in `mikro.ipynb`.
4. (Optional) Run the chatbot UI with Streamlit:
```bash
  streamlit run mikro.ipynb

```
5. (Optional) A version of code is also given to easily run on `Google Colab`
## UI Customization

The UI theme is configured via `config.toml`:
```toml
[theme]
primaryColor = "#4CAF50"
backgroundColor = "#F9F9F9"
secondaryBackgroundColor = "#E8EAF6"
textColor = "#262730"
font = "sans serif"
```
## Features

- LSTM-based intent classification
- Clean JSON-driven dataset
- Expandable with new intents and responses
- Optionally deployable with Streamlit
- Clear chat option


## Tech Stack

**Language:** 
  - Python — Core programming language for scripting, data handling, and model training

**Libraries & Frameworks:** 
 - TensorFlow / Keras — For building and training the LSTM neural network model
 - Scikit-learn — For preprocessing tasks (like Label Encoding and train-test split)
 - nltk (Natural Language Toolkit) — For tokenizing and lemmatizing user input
 - json — To load structured intent data for the chatbot
  - NumPy — Efficient numerical operations
  - Pandas (optional) — For data manipulation if needed
  - Streamlit — For building a simple, interactive web interface for the chatbot
  - TOML — For styling and configuring Streamlit themes


**Files:**
- `mikro.ipynb` — Model and logic notebook 
- `Mikro_LSTM_Dataset.json` — Custom dataset for intents and responses
- `config.toml` — Theme configuration for the UI
- `requirements.txt` — For easy installation of libraries


## Contributing

Contributions are always welcome!


