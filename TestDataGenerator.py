###################### GENERATE TEST DATA #########################
import string
import os
import random
from PIL import ImageDraw,ImageFont,Image
import pandas as pd
import datetime

os.chdir('/Users/ianbogley/Desktop/Data Science/image recognition/Image Character Reader')


log = pd.DataFrame({
    'set': [],
    'filename': [],
    'color': [],
    'font_file': [],
    'filepath': []
})
def GenerateData(char,dataset = str(datetime.date.today()),n=100):
    if char in string.ascii_lowercase:
        set = "lower"
    elif char in string.ascii_uppercase:
        set = "upper"
    elif char in string.digits:
        set = "digit"
    else:
        set = 'punct'
    date_dir = 'data/'+dataset
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
    for i in range(n):
        if i in (round(.25*n),round(.5*n),round(.75*n),n):
            print(char+': '+str(i/n*100)+'%')
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

characters = [char for char in string.ascii_letters+string.digits+string.punctuation]

#Generate directories for each character
if 'data' not in os.listdir(): 
    os.mkdir('data')

font_dir = '/Users/ianbogley/Desktop/Data Science/image recognition/Image Character Reader/fonts/'
font_population = os.listdir(font_dir)

log = pd.DataFrame({
    'set': [],
    'filename': [],
    'color': [],
    'font_file': [],
    'filepath': []
})

for char in characters:
    log=pd.concat([log,GenerateData(char,n=100)])
