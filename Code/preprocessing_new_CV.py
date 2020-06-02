from textblob import TextBlob
from googletrans import Translator
import unicodedata
import re
import spacy
nlp = spacy.load('en_core_web_md')


def remove_accent(word):
    return unicodedata.normalize('NFD', word).encode('ascii', 'ignore').decode("utf-8")


def translate_CV(CV):
    CV_traduit = None
    try :
        blob = TextBlob(CV )
        CV_traduit = str(blob.translate(from_lang='fr', to='en'))
    except:
        try :
            translator = Translator()
            CV_traduit = translator.translate( CV,src='fr', dest='en').text
        except :
            try:
                #decoupe le cv en 4 petits bouts
                lignes = CV.split('\n')
                nb_lignes = len(lignes)
                size = int(np.floor(nb_lignes/4))
                c1 = "\n".join(CV.split('\n')[0:size])
                c1_traduit = translator.translate(c1 ,src='fr', dest='en').text
                c2 = "\n".join(CV.split('\n')[size:2*size])
                c2_traduit = translator.translate(c2 ,src='fr', dest='en').text
                c3 = "\n".join(CV.split('\n')[2*size:3*size])
                c3_traduit = translator.translate(c3 ,src='fr', dest='en').text
                c4 = "\n".join(CV.split('\n')[3*size:])
                c4_traduit = translator.translate(c4 ,src='fr', dest='en').text
                cr = "\n".join([c1,c2,c3,c4])
                CV_traduit = "\n".join([c1_traduit, c2_traduit, c3_traduit, c4_traduit])
                
            except:
                print("la traduction du cv n'a pas fonctionné")
    return CV_traduit



def pre_processing(CV, vocab):
    #suppression des xxx
    CV_xx = re.sub("[x]{2,}|[X]{2,}", ' ', CV)
    
    #traduction en anglais
    CV_traduit = translate_CV(CV_xx)
    
    #suppression des caractères posant problèmes avec spacy
    CV_caract = re.sub("[())/><]|Social Security Number", ' ', CV_traduit) 
    
    #tokenization
    tokens = nlp(CV_caract)
    list_word_cv = [remove_accent(w.lemma_.lower()) for w in tokens if w.is_punct==False and w.is_space==False and not w.is_stop and not w.is_digit ]

    CV_join1 = " ".join(list_word_cv)
    
    #découpe le cv en mots en c++, c#
    CV_list = re.findall("c\+\+|c#|[0-9a-z]+" ,CV_join1)
    
    CV_vocab = [mot for mot in CV_list if mot in vocab]
    
    return CV_vocab
