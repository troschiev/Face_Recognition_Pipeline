import os
import io
import cv2
import PIL
import json
import torch
import numpy as np
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
from io import BytesIO
from torch import nn
from PIL import Image
from sklearn import metrics
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import normalize
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches

class FaceRegressionModel(nn.Module):
    def __init__(self, backbone, num_points=5, hidden_size=2**11):
        '''
        backbone - CNN for reature extraction
        num_points - number of predicted points (each point has two coordinates)
        hidden_size - number of neurons in hidden FC layer before output layer
        '''
        super(FaceRegressionModel, self).__init__()
        self.num_points = num_points
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Flatten(), # now the head can work with any conv result
            nn.LazyLinear(hidden_size),
            nn.LeakyReLU(0.05),
            nn.LazyLinear(num_points * 2)
        )

    def forward(self, tensor):
        tensor = self.backbone(tensor)
        tensor = self.head(tensor)
        tensor = tensor.view(tensor.shape[0], self.num_points, -1)
        return tensor
    
class FaceRecognitionModel(nn.Module):
    def __init__(self, backbone, embed_size=512, drop_rate=0.33):
        '''
        backbone - CNN for reature extraction
        num_classes - number of predicted classes (persons)
        embed_size - embeddind dimentions
        '''
        super(FaceRecognitionModel, self).__init__()
        self.embed_size = embed_size
        self.backbone = backbone
        
        self.embed = nn.Sequential(
            nn.Flatten(), # now the head can work with any conv result
            nn.Dropout(drop_rate), # regularization
            nn.LazyLinear(embed_size, bias=False), # embed!
            nn.LazyBatchNorm1d(), # to center embeddings around 0
        )

    def forward(self, tensor):
        tensor = self.backbone(tensor)
        tensor = self.embed(tensor)
        return tensor # embeddings
    
class FaceFinder:
    def __init__(self, 
                 detector_weights_path,
                 regressor_model_path,
                 regressor_config_path,
                 recognitor_model_path,
                 recognitor_config_path,
                 landmark_path,
                 custom_params_recognition_path,
                 embed_path,
                 limit_rotated_face=1.1,
                 limit_min_size=0.33,
                 max_faces_per_image=-1,
                 device='cpu'
                ):
        
        # load trained YOLO detector
        if device == 'cpu':
            kwarg = dict(device=device)
        else:
            kwarg = {}
        self.detector = torch.hub.load('ultralytics/yolov5',
                                       'custom', 
                                       path=detector_weights_path, 
                                       force_reload=True, 
                                       **kwarg
                                      )
        
        # load trained landmark coordinate regressor
        self.regressor = torch.load(regressor_model_path, 
                                    map_location=torch.device(device))
        self.regressor.eval()
        
        # load regressor config
        with open(regressor_config_path) as infile:
            regresson_cfg = json.load(infile)
        regresson_stats = regresson_cfg['mean'], regresson_cfg['std']
        regresson_img_size = regresson_cfg['input_size'][1]
        self.regressor_basic_transform = A.Compose([
                A.Resize(regresson_img_size, regresson_img_size),
                A.Normalize(*regresson_stats),
                ToTensorV2(),
            ])
        
        # load recognition model
        self.recognitor = torch.load(
            recognitor_model_path, 
            map_location=torch.device(device))
        self.recognitor.eval()
        
        # load recognition model config
        with open(recognitor_config_path) as infile:
            recognitor_cfg = json.load(infile)
        recognitor_stats = recognitor_cfg['mean'], recognitor_cfg['std']
        self.recognitor_img_size = recognitor_cfg['input_size']
        self.recognitor_basic_transform = A.Compose([
                A.Normalize(*recognitor_stats),
                ToTensorV2(),
            ])
        
        # load custom parameters for recognition alignment
        self.custom_params = json.load(open(custom_params_recognition_path, 'r'))
        
        # list of known images + embeddings
        self.df_full = pd.read_pickle(landmark_path)
        self.embeddings = np.load(embed_path)
        
        # alignment parameters
        self.desiredEyesY = self.custom_params['desiredEyesY']
        self.desiredFaceWidth = self.custom_params['im_width']
        self.desiredFaceHeight = self.custom_params['im_height']
        
        self.vertical_face_scale = self.custom_params['vertical_face_scale']
        self.limit_rotated_face = limit_rotated_face
        self.limit_min_size = limit_min_size
        
        # limit number of faces per image
        self.max_faces_per_image = max_faces_per_image
        
        self.device = device
        
    def landmarks_dist(self, coordinates):
    
        def pair_dist(xy1, xy2):
            return ((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)**0.5

        return dict(
            eye1eye2 = pair_dist(coordinates[0,:], coordinates[1,:]),
            eye1nose = pair_dist(coordinates[0,:], coordinates[2,:]),
            eye2nose = pair_dist(coordinates[1,:], coordinates[2,:]),
        )
    
    def align(self, image, coordinates, nose_weight=4):
        
        """
        Aligns image so that eyes are on a single horizontal line
        and scalse image by eye-to-eye distance OR by eyes_center-to-weighted_mouth+nose_center
        """

        leftEyeCenter = coordinates[0]
        rightEyeCenter = coordinates[1]
        
        # we will scale face by two sizes: 
        # eye-to-eye
        # eyes_center-to-weighted_lower_face_center
        mouth_weight = 1
        sum_weingt = 2 + nose_weight

        bottom_center = coordinates[2] * nose_weight + (coordinates[3] + coordinates[4])
        bottom_center = bottom_center / sum_weingt

        top_center = (leftEyeCenter + rightEyeCenter) / 2

        # compute the angle between the eye centroids
        d_eyes = rightEyeCenter - leftEyeCenter
        angle = np.degrees(np.arctan2(*d_eyes[::-1]))

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        # desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        top_bottom_dist = (np.abs(
            np.linalg.norm(
                np.cross(
                    rightEyeCenter-leftEyeCenter, 
                    leftEyeCenter-bottom_center
                )
            )
        ) / np.linalg.norm(
            rightEyeCenter-leftEyeCenter
        ))

        eyes_dist = np.linalg.norm(d_eyes)

        size_y = self.desiredFaceHeight
        scale1 = size_y / (top_bottom_dist * self.vertical_face_scale)
        scale2 = size_y / (eyes_dist * self.vertical_face_scale / 1.29)

        # scale image to get same distance between eyes OR between top and bottom "face centers"
        scale = min(scale1, scale2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(top_center, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredEyesY
        M[0, 2] += (tX - top_center[0])
        M[1, 2] += (tY - top_center[1])

        # apply the affine transformation
        output = cv2.warpAffine(src=image, 
                                M=M, 
                                dsize=(
                                    self.desiredFaceWidth, 
                                    self.desiredFaceHeight
                                ),
                                flags=cv2.INTER_CUBIC)

        return output    
        
    def detect_bboxes(self, img):
        img = np.array(img)
        bboxes = self.detector(img)
        bboxes = bboxes.xyxy[0].cpu().numpy()[:,:-2].astype(int)
        return bboxes, img
    
    def extract_landmarks_coordinates(self, face_cut, bbox):
        face_cut = self.regressor_basic_transform(image=face_cut)['image']
        face_cut = face_cut.unsqueeze(0)
        with torch.no_grad():
            face_cut = face_cut.to(self.device)
            coordinates = self.regressor(face_cut)
            coordinates = coordinates.cpu()
        coordinates = coordinates.squeeze(0).detach().numpy()

        coordinates[:,0] = (bbox[2]+bbox[0])/2 + coordinates[:,0] * (bbox[2]-bbox[0])
        coordinates[:,1] = (bbox[3]+bbox[1])/2 + coordinates[:,1] * (bbox[3]-bbox[1])
        
        return coordinates
    
    def get_cut_image(self, bbox, img):
        x1 = int(bbox[0])
        x2 = int(bbox[2])
        y1 = int(bbox[1])
        y2 = int(bbox[3])
        img = img[y1:y2, x1:x2]
        return img, min(
            abs(x1-x2),
            abs(y1-y2)
        )
    
    def check_rotated_face(self, coordinates):
        face_metr = self.landmarks_dist(coordinates)
        max_eye1nose = ((face_metr['eye1eye2'])**2+(face_metr['eye2nose'])**2)**0.5
        max_eye2nose = ((face_metr['eye1eye2'])**2+(face_metr['eye1nose'])**2)**0.5
        b1 = face_metr['eye1nose'] > max_eye1nose*self.limit_rotated_face 
        b2 = face_metr['eye2nose'] > max_eye2nose*self.limit_rotated_face
        return b1 or b2
    
    def get_face_embedding(self, img):
        img = self.recognitor_basic_transform(image=img)['image']
        img = img.unsqueeze(0)
        with torch.no_grad():
            img = img.to(self.device)
            embed = self.recognitor(img)
            embed = embed.cpu()
        embed = embed.squeeze(0).detach().numpy()
        return embed
    
    def get_cos_similarity(self, emb_in):
        sim_array= metrics.pairwise.cosine_similarity(
            self.embeddings, emb_in[None, :]
        ).squeeze()
        return sim_array
    
    def PIL_autocontrast_numpy(self, img):
        return np.array(
                PIL.ImageOps.autocontrast(
                    Image.fromarray(img)
                )
            )
    
    def get_faces_from_df(self, similarity_list, num_similar=8, quant=0.):
        
        df_small = self.df_full[self.df_full.columns]
        df_small['similarity'] = similarity_list
        
        if quant > 0:
            w_min = df_small['w'].quantile(quant)
            h_min = df_small['h'].quantile(quant)
            df_small = df_small[df_small['w'] >= w_min]
            df_small = df_small[df_small['h'] >= h_min]
            
        df_small['max_sim'] = df_small.groupby(
            'person'
        )['similarity'].transform('max')
        df_small = df_small[df_small['similarity'] == df_small['max_sim']]
        df_small = df_small.sort_values('similarity', ascending=False)
        
        images = []
        sims = []
        for i, row in df_small.head(num_similar).iterrows():
            sim = row['similarity']
            fname = row['file']
            image = cv2.imread(fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            coords = row['lm_abs']
            image = self.align(image, coords)
            images.append(image)
            sims.append(sim)
            
        return images, sims
    
    def create_face_table(self, target_image, similar_images_list, sims):
    
        hw = np.array([3.7, 5])
        fig = plt.figure(figsize=hw*1.5)

        l = len(similar_images_list)
        # full_list = similar_images_list[:l//2] + [target_image] + similar_images_list[l//2:]
        full_list = [target_image] + similar_images_list
        sims = [1] + sims
        
        min_sim = 0.6
        linewidth = 5
        text_offset = 5

        grid_size = int((l+1)**0.5)

        for i in range(grid_size):
            for j in range(grid_size):
                k = i*grid_size + j
                plt.subplot(grid_size, grid_size, k + 1)
                image = full_list[k]
                sim = sims[k]
                g = max(0., (sim - min_sim) / (1. - min_sim))
                r = 1. - g
                rgb = np.array((r,g,0)) * 255.
                image = cv2.copyMakeBorder(image,
                                           linewidth,linewidth,linewidth,linewidth,
                                           cv2.BORDER_CONSTANT,
                                           value=rgb)
                
                if k == 0:
                    title = 'Original'
                else:
                    title = f'{sim:.2f}'
                
                x = image.shape[1] - linewidth - text_offset
                y = image.shape[0] - linewidth - text_offset
                
                
                plt.imshow(image)
                plt.axis('off')
                
                text = plt.text(x, y, 
                                title, color='white',
                                ha='right', va='bottom', size=25)

                text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
                                       path_effects.Normal()])
                text.set_in_layout(False)
                

        fig.patch.set_facecolor('xkcd:white')
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        return PIL.Image.open(buf)
    
    def find_faces(self, img, num_similar=3**2-1):
        
        bboxes, img = self.detect_bboxes(img)
        
        faces = []
        for bbox_idx, bbox in enumerate(bboxes):
            
            message='found face'
            
            if self.max_faces_per_image > 0:
                if box_idx >= self.max_faces_per_image:
                    break
            
            face_cut, min_size = self.get_cut_image(bbox, img)
            coordinates = self.extract_landmarks_coordinates(face_cut, bbox)
            
            face_cut = self.align(image=img, coordinates=coordinates)
            
            if self.limit_min_size > 0:
                if min_size < self.limit_min_size * self.desiredFaceWidth:
                    message='small face'
                    basewidth = 100
                    img = Image.fromarray(face_cut)
                    wpercent = (basewidth/float(img.size[0]))
                    hsize = int((float(img.size[1])*float(wpercent)))
                    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
                   
                    faces.append(dict(
                        message=message,
                        image=img
                    ))
                    continue
                    
            if self.limit_rotated_face > 0:
                if self.check_rotated_face(coordinates):
                    message='rotated face'
                    faces.append(dict(
                        message=message,
                        image=Image.fromarray(face_cut)
                    ))
                    continue
                    
            
            face_cut = self.PIL_autocontrast_numpy(face_cut)
            embed = self.get_face_embedding(face_cut)
            similarity_list = self.get_cos_similarity(embed)
            similar_images_list, sims = self.get_faces_from_df(similarity_list, num_similar=num_similar)
            face_table = self.create_face_table(face_cut, similar_images_list, sims)
            
            faces.append(dict(
                        message=message,
                        image=face_table,
                    ))
        
        return faces