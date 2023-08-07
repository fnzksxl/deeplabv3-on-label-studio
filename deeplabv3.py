import torch
import os
import logging
import boto3
import io
import json
import requests
import config
import deeplab_resnet

from run_train import train
from dataset import make_dataset
from utils import *
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, \
    get_single_tag_keys, DATA_UNDEFINED_NAME, get_env
from label_studio_tools.core.utils.io import get_data_dir
from label_studio_converter import brush

from urllib.parse import urlparse
from botocore.exceptions import ClientError
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import transforms

logger = logging.getLogger(__name__)

HOSTNAME = get_env('HOSTNAME', 'http://localhost:8080')
API_KEY = get_env('API_KEY')

print('=> LABEL STUDIO HOSTNAME = ', HOSTNAME)
if not API_KEY:
    print('=> WARNING! API_KEY is not set')

device = "cuda" if torch.cuda.is_available() else "cpu"

class DEEPLABV3(LabelStudioMLBase):
    def __init__(self,labels_file=None, device=device, **kwargs):
        super(DEEPLABV3,self).__init__(**kwargs)

        # self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
        #         self.parsed_label_config, 'PolygonLabels', 'Image')
        
        schema = list(self.parsed_label_config.values())
        self.to_name = schema[0]['to_name'][0]
        self.label_length = len(schema[0]['labels'])

        # Mask Color Map 생성
        self.brush_color_map = {}
        for s in schema:
            if s['type'] == 'BrushLabels':
                attrs=s['labels_attrs']
                for key in attrs.keys():
                    self.brush_color_map[key] = attrs[key]['background']

        # Mask 와 Polygon Label 이름, 순서가 같다는 전제
        self.label_map = {i:k for i,k in enumerate(schema[1]['labels'])}
        self.label_map_reversed = {k:i for i,k in self.label_map.items()}
        self.value = schema[0]['inputs'][0]['value']
        
        print(self.label_map)
        print(self.label_map_reversed)

        if os.path.exists('./saved_model/best.pth'):
            # Background도 class에 포함시켜야 해서 +1 해줌
            self.model = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=self.label_length+1, os=16).to(device)
            self.model.load_state_dict(torch.load('./saved_model/best.pth'))
            self.mode = 'customized'
            print("Using Customized Model.")
        else:
            self.model = deeplabv3_resnet101(weights='DeepLabV3_ResNet101_Weights.DEFAULT').to(device)
            self.mode = 'pretrained'
            print("Using Pretrained Model.")

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3',region_name="ap-northeast-2", config=Config(signature_version="s3v4"))
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key,
                            'ResponseContentDisposition': 'attachment'},
                    ExpiresIn = 10000,
                    
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
            
            return image_url, key
        return image_url    


    def predict(self, tasks, **kwargs):
        prompt_type = kwargs['context']
        print(f"kwargs: {kwargs}")
        print(f"prmpt_type: {prompt_type}")
        if prompt_type != None:
            return []
        
        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = [self.get_local_path("http://label-studio:8080" + image_url)]
        
        # Predict Sequence
        image = Image.open(image_path[0]).convert('RGB')
        image_size = image.size
        input_tensor = self.preprocess(image).unsqueeze(0).to(device)

        self.model.eval()
        # 같은 resnet-101 인데 살펴보니 레이어 구조가 달라서, 구분해줌
        if self.mode == 'pretrained':
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
        elif self.mode == 'customized':
            with torch.no_grad():
                output = self.model(input_tensor)[0]
       
       
        results = []
        mask = output.argmax(0).cpu().numpy()
        object_masks = return_mask_label_name(mask,self.label_map)

        print("Object Masks >>> "+str(len(object_masks)))
        for one_mask,label in object_masks:
            label_name = self.label_map[label-1]
            encoded_mask = brush.mask2rle(one_mask*255)
            results.append({
                'from_name' : 'tag',
                'to_name' : self.to_name,
                'type' : 'brushlabels',
                'value' : {
                    'format' : 'rle',
                    'rle' : encoded_mask,
                    'brushlabels' : [
                        label_name
                    ]
                }
            })
        polygonlabel = mask_to_polygons(mask)
        print("PolygonLabel length >>> ")
        print(len(polygonlabel))
        for poly, label_num in polygonlabel:
            normal_polygons = normalize_polygons(poly,image_size)
            # Predict result upload Sequence
            results.append({
                    'from_name': 'label',
                    'to_name': self.to_name,
                    'type': 'polygonlabels',
                    'value': {
                            'points' : normal_polygons,
                            'polygonlabels' : [self.label_map[label_num-1]]
                    },
        
                })
        
        return [{
            'result': results
        }]
    
    def _get_annotated_dataset(self, project_id):
        """Just for demo purposes: retrieve annotated data from Label Studio API"""
        self.download_url = f'{HOSTNAME.rstrip("/")}/api/projects/{project_id}/export'
        response = requests.get(self.download_url, headers={'Authorization': f'Token {API_KEY}'})
        if response.status_code != 200:
            raise Exception(f"Can't load task data using {self.download_url}, "
                            f"response status_code = {response.status_code}")
        return json.loads(response.content)
    
    def fit(self, annotations, workdir=None, batch_size=2, num_epochs=10, **kwargs):
        print('Collecting annotations...')        

        # check if training is from web hook
        if kwargs.get('data'):
            project_id = kwargs['data']['project']['id']
            tasks = self._get_annotated_dataset(project_id)
            
        # ML training without web hook
        else:
            tasks = annotations

        # Make Dataset before Train
        image_path_lst = []
        label_mask_lst = []
        for task in tasks:
            if not task.get('annotations'):
                continue
            annotation = task['annotations'][0]
            
            if annotation.get('was_cancelled') or annotation.get('skipped'):
                continue
            
            image_url = self._get_image_url(task)
            image_path = '/app'+image_url
            image_size=get_image_size(image_path)

            results = annotation['result']
            mask_lst = []
            for result in results:
                if result['from_name'] != 'label':
                    continue
                polygonlabel = result['value']['points']
                label = result['value']['polygonlabels'][0]
                label_num = self.label_map_reversed[label]
                mask_lst.append(create_mask_from_polygons(polygonlabel,label_num,image_size))
            
            # SegmentationClass 마스크
            merged_mask = np.logical_or.reduce(mask_lst)

            image_path_lst.append(image_path)
            label_mask_lst.append(merged_mask)

        train_val_rate = config.train_val_rate 
        ceil = int(len(image_path_lst)*train_val_rate)

        train_image = image_path_lst[:ceil]
        val_image = image_path_lst[ceil:]
        train_label = label_mask_lst[:ceil]
        val_label = label_mask_lst[ceil:]
        
        traindata = make_dataset(train_image,train_label,'train',config.composed_transforms_tr)
        valdata = make_dataset(val_image,val_label,'val',config.composed_transforms_ts)

        train(traindata,valdata,self.model,self.mode,self.label_length+1)


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data