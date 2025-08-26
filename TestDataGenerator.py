###################### GENERATE TEST DATA #########################
import string
import os
import random
from PIL import ImageDraw,ImageFont,Image
import pandas as pd
import datetime
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor




#Get names of punctuation marks
punctuation_data = [
    (".", "Period / Full Stop"),
    (",", "Comma"),
    (";", "Semicolon"),
    (":", "Colon"),
    ("?", "Question Mark"),
    ("!", "Exclamation Mark"),
    ("-", "Hyphen"),
    ("–", "En Dash"),
    ("—", "Em Dash"),
    ("_", "Underscore"),
    ("(", "Left Parenthesis"),
    (")", "Right Parenthesis"),
    ("[", "Left Square Bracket"),
    ("]", "Right Square Bracket"),
    ("{", "Left Curly Brace"),
    ("}", "Right Curly Brace"),
    ("'", "Apostrophe"),
    ("\"", "Quotation Mark"),
    ("‘", "Left Single Quotation Mark"),
    ("’", "Right Single Quotation Mark"),
    ("“", "Left Double Quotation Mark"),
    ("”", "Right Double Quotation Mark"),
    ("/", "Slash / Forward Slash"),
    ("\\", "Backslash"),
    ("|", "Vertical Bar / Pipe"),
    ("@", "At Symbol"),
    ("#", "Hash / Pound / Number Sign"),
    ("$", "Dollar Sign"),
    ("%", "Percent Sign"),
    ("^", "Caret"),
    ("&", "Ampersand"),
    ("*", "Asterisk"),
    ("+", "Plus Sign"),
    ("=", "Equals Sign"),
    ("<", "Less Than Sign"),
    (">", "Greater Than Sign"),
    ("~", "Tilde"),
    ("`", "Grave Accent"),
]

# Format punctuation data for matching
punctuation_names = pd.DataFrame(punctuation_data, columns=["Punctuation", "Name"])
punctuation_names['Name'] = punctuation_names['Name'].str.replace(' ',''). \
    str.replace('/','').str.replace('\\','').str.replace('-','').str.replace('(','') \
        .str.replace(')','').str.replace('[','').str.replace(']','').str.replace('{','') \
            .str.replace('}','')

log = pd.DataFrame({
    'set': [],
    'filename': [],
    'color': [],
    'font_file': [],
    'filepath': []
})

#Generate directories for each character
if 'data' not in os.listdir(): 
    os.mkdir('data')


#Define data generating function
def GenerateData(char,n=100):
    #Create directory for current data generation
    dataset = str(datetime.date.today())
    date_dir = 'data/'+dataset
    if char in string.ascii_lowercase:
        set = "lower"
    elif char in string.ascii_uppercase:
        set = "upper"
    elif char in string.digits:
        set = "digit"
    else:
        set = 'punct'
    #Create directory for current character
    if set == 'punct':
        char_name = punctuation_names[punctuation_names['Punctuation']==char]['Name'].values[0]
        char_dir = date_dir+'/'+char_name+'_'+set
    else:
        char_dir = date_dir+'/'+char+'_'+set

    if dataset not in os.listdir('data'):
        print('making '+date_dir)
        os.mkdir(date_dir)
    os.mkdir(char_dir)
    
    char_log = pd.DataFrame({
        'set': [],
        'filename': [],
        'color': [],
        'font_file': [],
        'filepath': []
    })
    font_dir = 'fonts/'
    font_population = os.listdir(font_dir)

    #Generate the data
    for i in range(n):

        color = (random.randint(1,255),random.randint(1,255),random.randint(1,255))
        
        font_file = font_dir+font_population[random.randint(0,len(font_population)-1)]
        font = ImageFont.truetype(font_file,random.randint(10,50))

        img = Image.new('RGB',(100,100),color=color)

        ImageDraw.Draw(img).text((30,15),char,font=font,fill=(0,0,0))
        
        img.save(char_dir+'/img'+str(i)+'.png')
        templog = pd.DataFrame({
                'set': [set],
                'filename': ['img'+str(i)+'.png'],
                'color': [color],
                'font_file': [font_file],
                'filepath': [char_dir+'/img'+str(i)+'.png']
            })
        char_log = pd.concat([char_log,templog])
    return char_log

#Generate character images in parallel
""" def parallel_generate(characters, n=10000):
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(GenerateData, c, n): c for c in characters}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Data"):
            results.append(future.result())
    return results """
def parallel_generate(characters, n=100):
    results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(GenerateData, c, n): c for c in characters}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Data"):
            results.append(future.result())
    return results

if __name__ == "__main__":
    characters = [char for char in string.ascii_letters+string.digits+string.punctuation]
    results = parallel_generate(characters, n=10000)
    print("Done!")


#log = pd.DataFrame({
#    'set': [],
#    'filename': [],
#    'color': [],
#    'font_file': [],
#    'filepath': []
#})

#starttime = datetime.datetime.now()
#for char in characters:
#    log=pd.concat([log,GenerateData(char,n=10000)])
#endtime = datetime.datetime.now()

#Run parallel process
""" if __name__ == "__main__":
    characters = [char for char in string.ascii_letters+string.digits+string.punctuation]

    results = parallel_generate(characters, n=10000, max_workers=8)
    print("\nAll data generation complete!")
    print(results)
for i in characters:
    print(i)
    GenerateData(i,n=1) """