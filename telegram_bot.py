from custom_models import FaceRegressionModel, FaceRecognitionModel, FaceFinder

import uuid
import torch
import time
import requests
import PIL
import pandas as pd
import os
import numpy as np
import nest_asyncio
import matplotlib.pyplot as plt
import logging
import json
import io
import cv2
import albumentations as A
from torch import nn
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from PIL import Image
from io import BytesIO
from albumentations.pytorch import ToTensorV2
from aiogram import Bot, Dispatcher, executor, types
nest_asyncio.apply()

verb = False

def print_v(*text):
    if verb: print(*text)
    
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

if 'nt' in os.name:
    path_file = 'path_windows.txt'
else:
    path_file = 'path_linux.txt'

tmp_dir = 'tmp/'
path_to_token = open(path_file, 'r')
path_to_token = path_to_token.readlines()[0].rstrip('\n')
path_to_token = open(path_to_token, 'r')
API_TOKEN = path_to_token.readlines()[0].rstrip('\n')

face_finder = FaceFinder(
    detector_weights_path='./models/detection/best.pt',
    regressor_model_path='./models/landmark/best_landmark_model.pt',
    regressor_config_path='./models/landmark/best_landmark_model_cfg.txt',
    recognitor_model_path='./models/recognition/best_recog_model_arc.pt',
    recognitor_config_path='./models/recognition/best_model_cfg.txt',
    landmark_path='./tables/final_landmarks_small.pkl',
    embed_path='./tables/embed_arc.npy',
    custom_params_recognition_path='./models/recognition/custom_params.txt',
    device='cpu'
)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start', 'help'])
async def start(message: types.Message):
    print_v('received start command')
    text = 'Hi! Please send photo with some faces.'
    text += ' This bot will find some celebrities that look like people on the photo!'
    await bot.send_message(message.chat.id, text=text)

@dp.message_handler(content_types=types.ContentTypes.TEXT)
async def handle_text_message(message: types.Message):
    print_v('text received')
    chat_id = message.chat.id
    print_v('chat_id', chat_id)
    text = message.text
    print_v('message.text', text)
    if text:
        print_v('replying')
        await bot.send_message(chat_id=chat_id, text='Please send photo with some faces!')
        
@dp.message_handler(content_types=types.ContentTypes.PHOTO)
async def handle_docs_photo(message: types.Message):
    # get almost unique value
    unique_id = uuid.uuid4().hex
    print_v('photo received')
    
    # create file name
    chat_id = message.chat.id
    fname = tmp_dir + str(chat_id) + '_' + unique_id + '.jpg'
    print_v(fname)
    
    # download and save photo
    await message.photo[-1].download(destination_file=fname)
    print_v('photo saved')
    
    # read photo
    image = cv2.imread(fname)
    print_v('photo read')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print_v('photo converted')
    
    # find faces, save them to results variable
    results = face_finder.find_faces(img=image)
    print_v('faces found')
    print_v(len(results))
    
    # send message with number of faces
    if len(results) > 1:
        text = f'Found {len(results)} faces:'
        await bot.send_message(chat_id=chat_id, text=text)
    elif len(results) == 0:
        text = f'No faces found!'
        await bot.send_message(chat_id=chat_id, text=text)
    
    for i, r in enumerate(results):
        # new temp file name
        unique_id = uuid.uuid4().hex
        fname_res = tmp_dir + str(chat_id) + '_' + unique_id + '_res' + str(i) + '.png'
        
        # send message with face number
        text = ''
        if len(results) > 1:
            text = f'Face #{i+1} '
        
        print_v(r['message'])
        # send "bad face" message
        if r['message'] == 'rotated face':
            text += 'the face is turned too much:'
        elif r['message'] == 'small face':
            text += 'the face is too small:'
        print_v(text)
        if len(text) > 0:
            await bot.send_message(chat_id=chat_id, 
                                   text=text)
        
        # show image
        if r['image'] != None:
            r['image'].save(fname_res)
            print_v('tmp file saved')
            with open(fname_res, 'rb') as f_open:
                await bot.send_photo(chat_id=chat_id, photo=f_open)
            print_v('image sent')
            
            # delete temp file
            os.remove(fname_res)
            print_v('tmp file deleted')
            
    os.remove(fname)
    
try: os.mkdir('tmp')
except: pass
    
executor.start_polling(dp, skip_updates=True)