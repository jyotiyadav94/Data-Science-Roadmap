from datasets.dataset_orderSentinel3Products import data
from collections import defaultdict
from datetime import datetime

class OrderSentinel3Products:
    def __init__(self, data):
        self.data = data

    def process_data(self):
        grouped_images = defaultdict(list)
        for image in self.data:
            data_type_id = image.split('_')[3]
            grouped_images[data_type_id].append(image)

        for data_type_id, images in grouped_images.items():
            images.sort(key=lambda x: datetime.strptime(x.split('_')[7], '%Y%m%dT%H%M%S'))

        return grouped_images

# Assuming 'data' is imported correctly from datasets.dataset_orderSentinel3Products
data_processor = OrderSentinel3Products(data)
grouped_and_ordered_images = data_processor.process_data()

# printing grouped and ordered images
for data_type_id, images in grouped_and_ordered_images.items():
    print(f'Data Type ID: {data_type_id}')
    for image in images:
        print(image)
    print()
