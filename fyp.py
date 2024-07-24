# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:16:55 2024

@author: Hp Elitebook 1040 G3
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("dammyogt/damilola-finetuned-kde4-en-to-yo")
model = AutoModelForSeq2SeqLM.from_pretrained("dammyogt/damilola-finetuned-kde4-en-to-yo")


# Title of the app
st.title('English to Yoruba Translator')

# Input text box
st.write('### TRANSCRIPT: ENGLISH')
english_text = st.text_area('Enter English text here:', '')

# Initialize Yoruba translation text area
yoruba_text = ''

# Translate button
if st.button('Translate'):
    # Perform translation
    a = model.generate(**tokenizer.prepare_seq2seq_batch(english_text, return_tensors="pt"))
    yoruba_text = tokenizer.batch_decode(a)

if type(yoruba_text)==list:
    yoruba_text=yoruba_text[0].replace("<pad>", "").replace("</s>", "").strip()

# Display translation
st.write('### TRANSLATION: YORUBA')
st.text_area('Yoruba Translation:', yoruba_text, height=100)
