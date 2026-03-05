import torch
import jiwer
from jiwer import transforms, wer

sub_dict = {"de":"dom",
            "dem":"dom",}

transform = jiwer.Compose([
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.SubstituteWords(sub_dict),
    ])


def spacy_segmentation(corpus, spacy_pipeline):
    doc = spacy_pipeline(corpus)
    sentences = []
    for sent in doc.sents:
        sentences.append(sent.text)
    return sentences


def get_sentence_embedding(text, embed_tokenizer, embed_model):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().numpy()


def analyze_speech(sentence_list, stz_nlp):
    words = []
    pos_list = []
    for s in sentence_list:
        doc = stz_nlp(s)
    
        for sentence in doc.sentences:
            for word in sentence.words:
                pos = word.pos
                if pos != "PUNCT":
                    words.append(word.text)
                    pos_list.append(word.pos)

    if not words:
      print("Children's speech was not recognized properly by the model.")

    else:
      adj_ratio = len([i for i in pos_list if i=="ADJ"])/len(words)
      ttr = len(list(set(words)))/len(words)
      
      kids_speech = ' '.join(sentence_list)
      print("Kid's Speech:")
      print(kids_speech)
      print(f"[Type-Token Ratio (Corpus)]: {ttr:.2f}")
      print(f"[Adjective Ratio]: {adj_ratio:.2f}")
    
    print("-----\n")

