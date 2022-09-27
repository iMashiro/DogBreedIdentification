import os

from torchvision import transforms

from PIL import Image

class ProcessDataset():
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_count = 1

    def generate_label_columns(self, columns, label):
        result = []
        for breed in columns:
            if breed == label:
                result.append(1.0)
            else:
                result.append(0.0)
        return result

    def image_process(self, data, data_type, columns):
        print('Processing image ' + str(self.image_count))

        image_id = data['id']
        if data_type == 'train':
            image_label = data['breed']

        image_path = self.data_path + data_type + '/' + image_id + '.jpg'

        image = Image.open(image_path)

        pipeline = transforms.Compose([transforms.Resize(64),
                                        transforms.CenterCrop(64),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        image = pipeline(image)

        if data_type == 'train':
            labels = self.generate_label_columns(columns, image_label)
        else:
            labels = [data[column] for column in columns]

        self.image_count += 1
        return [image] + labels
