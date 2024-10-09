---
title: BERT Emotion Detector
emoji: 🏃
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Input a random sequence of Text and get an Emotion as output
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
# BERT_Sentiment_Analysis_Bot

## 🔦Introduction
Bilions and billions of messages are sent across social media platforms all the time.

Sometimes it is important to understand the message and the tone that it's implying in order to produce a correct response.

Welcome to my bot, which takes in a sentence and outputs a valid emotion :)

## 🔧What I implemented?
I wanted a program which could recognise a sentence and determined what it meant, so what better model to use than the infamous BERT language model!

A classic sentiment_analysis problem

Thus, I utilized a pretrained BERT model and trained it on a large set of text-based emotions found on <a href="https://www.kaggle.com/datasets/ishantjuyal/emotions-in-text/data">Kaggle</a> and it achieved test accuracy as high as 90%

## 🚧Challenges faced

This was one of my first few NLP projects thus, learning how to extract data and producing the training and testing loops are different to what I'm accustomed to.

Most of my PyTorch projects are in Image classification. Thus, I had to learn functions like how to tokenise sentences, attention_masks and much more.

This was certainly a tough feat, but after a lot of StackOverflow and time spent on my computer, things start to fall into place

## 💡Future Implementations

This dataset which I extracted from Kaggle is not the mot valid dataset out there to determine emotons by sentence 

For example sentences like ```butterflies``` or ```butterflies in my stomach``` is projected as ```Fear``` insted of ```Love``` 

Thus, my future implementation would involve me in finding or making a better training dataset and maybe utilising Instagram API to be able to authenticate messages sent and its valid emotion
## 👀Live Preview

Here it is! As shown in <a href="https://huggingface.co/spaces/Moonshallow5/BERT_Emotion_Detector">HuggingFace</a>
