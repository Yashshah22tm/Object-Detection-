{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMYaH8kJwoT7H2sDy5Z5VbV",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Yashshah22tm/Object-Detection-/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cdn-DkNxF0JV"
      },
      "outputs": [],
      "source": [
        "import streamlit as st\n",
        "import numpy as np\n",
        "import pickle\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.models import Model\n",
        "from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input\n",
        "from tensorflow.keras.preprocessing.image import load_img, img_to_array\n",
        "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
        "\n",
        "# Load MobileNetV2 model\n",
        "mobilenet_model = MobileNetV2(weights=\"imagenet\")\n",
        "mobilenet_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)\n",
        "\n",
        "# Load your trained model\n",
        "model = tf.keras.models.load_model('mymodel.h5')\n",
        "\n",
        "# Load the tokenizer\n",
        "with open('tokenizer.pkl', 'rb') as tokenizer_file:\n",
        "    tokenizer = pickle.load(tokenizer_file)\n",
        "\n",
        "# Set custom web page title\n",
        "st.set_page_config(page_title=\"Caption Generator App\", page_icon=\"📷\")\n",
        "\n",
        "# Streamlit app\n",
        "st.title(\"Image Caption Generator\")\n",
        "st.markdown(\n",
        "    \"Upload an image, and this app will generate a caption for it using a trained LSTM model.\"\n",
        ")\n",
        "\n",
        "# Upload image\n",
        "uploaded_image = st.file_uploader(\"Choose an image\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
        "\n",
        "# Process uploaded image\n",
        "if uploaded_image is not None:\n",
        "    st.subheader(\"Uploaded Image\")\n",
        "    st.image(uploaded_image, caption=\"Uploaded Image\", use_column_width=True)\n",
        "\n",
        "    st.subheader(\"Generated Caption\")\n",
        "    # Display loading spinner while processing\n",
        "    with st.spinner(\"Generating caption...\"):\n",
        "        # Load image\n",
        "        image = load_img(uploaded_image, target_size=(224, 224))\n",
        "        image = img_to_array(image)\n",
        "        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))\n",
        "        image = preprocess_input(image)\n",
        "\n",
        "        # Extract features using VGG16\n",
        "        image_features = mobilenet_model.predict(image, verbose=0)\n",
        "\n",
        "        # Max caption length\n",
        "        max_caption_length = 34\n",
        "\n",
        "        # Define function to get word from index\n",
        "        def get_word_from_index(index, tokenizer):\n",
        "            return next(\n",
        "                (word for word, idx in tokenizer.word_index.items() if idx == index), None\n",
        "        )\n",
        "\n",
        "        # Generate caption using the model\n",
        "        def predict_caption(model, image_features, tokenizer, max_caption_length):\n",
        "            caption = \"startseq\"\n",
        "            for _ in range(max_caption_length):\n",
        "                sequence = tokenizer.texts_to_sequences([caption])[0]\n",
        "                sequence = pad_sequences([sequence], maxlen=max_caption_length)\n",
        "                yhat = model.predict([image_features, sequence], verbose=0)\n",
        "                predicted_index = np.argmax(yhat)\n",
        "                predicted_word = get_word_from_index(predicted_index, tokenizer)\n",
        "                caption += \" \" + predicted_word\n",
        "                if predicted_word is None or predicted_word == \"endseq\":\n",
        "                    break\n",
        "            return caption\n",
        "\n",
        "        # Generate caption\n",
        "        generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)\n",
        "\n",
        "        # Remove startseq and endseq\n",
        "        generated_caption = generated_caption.replace(\"startseq\", \"\").replace(\"endseq\", \"\")\n",
        "\n",
        "    # Display the generated caption with custom styling\n",
        "    st.markdown(\n",
        "        f'<div style=\"border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;\">'\n",
        "        f'<p style=\"font-style: italic;\">“{generated_caption}”</p>'\n",
        "        f'</div>',\n",
        "        unsafe_allow_html=True\n",
        "    )"
      ]
    }
  ]
}