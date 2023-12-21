import re

def clean_text(text):
    text = re.sub(r"^[А-ЯЁ].{2,}\W{2}РИА Новости, [\w\s,]+\.", "", text)
    text = re.sub(r"^[А-ЯЁ].{2,}\W{2}РИА Новости\.", "", text)
    text = re.sub(r'http[^А-Я]+', ' ', text)
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r'ё', 'е', text)        
    return text.strip()

def preprocess_text(df, only_titles=False):
    df['content_clean'] = df.content.apply(clean_text)
    df['title_clean'] = df.title.apply(clean_text)
    if only_titles:  
    	df['text_clean'] = df['title_clean']
    else:
        df['text_clean'] = df['title_clean'] + ' ' + df['content_clean']

    return df