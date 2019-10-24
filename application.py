from flask import Flask, request, jsonify
import traceback
import pandas as pd
import numpy as np
import json
from rake_nltk import Rake
import re
import nltk
from fuzzywuzzy import fuzz
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)


@app.route('/', methods=['POST'])

def topic():
    try:
        #Obtengo datos de request
        jsons = request.get_json()
        
        #Convierto request a DF
        query_df = pd.DataFrame(jsons)
    
        #Convierto a DF el input de topicos y keywords
        topicos=pd.DataFrame(query_df['input'][0])
        
        #Pongo el lowercase 
        topicos['keywords']=topicos['keywords'].str.lower()
        
        #Configuro Rake para español con minimo de palabras de 1 y 3 para encontrar posibles n-grams
        r=Rake(language='spanish',min_length=1, max_length=3)
        
        #Convierto texto a DF y quito caracteres especiales, y pongo en minuscula
        text=pd.DataFrame(query_df['text'])
        texto=text['text'][0]

        texto = texto.replace('\n', '')
        texto = texto.replace('\"', '')
        texto = texto.replace('"', '')
        texto = texto.replace('#', '')
        texto = texto.replace('%', '')
        texto = re.sub(r"http\S+", '', texto)
        texto = re.sub(r"@\S+", '', texto)
        texto= texto.lower()
        
        #Extraigo Keywords
        r.extract_keywords_from_text(texto)
        kwords=r.get_ranked_phrases_with_scores()  
        #print(kwords)
        
        #Preparo cadena de texto para comparar
        postkeywords=""
        for i in range(0,len(kwords)):
            postkeywords+=kwords[i][1]+"|"
        
        #Comparacion topicos/keywords con keywords obtenidos del post
        topicos['score']=0
        for i in range(0,topicos.shape[0]):
            topicos['score'][i] = fuzz.token_set_ratio(topicos['keywords'][i],postkeywords)
        print(topicos)
        
        #return jsonify(topicos.to_json(orient='records'))
        return (topicos.to_json(orient='records'))
        #return topicos.to_json(orient='records')
    except:
        return jsonify({'trace': traceback.format_exc()})
        #return print('no funciono')

if __name__ == '__main__':
    app.run(debug=True)