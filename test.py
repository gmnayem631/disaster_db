
import spacy
nlp = spacy.load('models/trained_ner_v2/model-best')
text = 'The flood situation has worsened in Sylhet district killing 12 people.'
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
