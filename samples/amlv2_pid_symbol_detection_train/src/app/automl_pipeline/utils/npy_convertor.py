import cv2


class NpyConvertor:
    """Converting Npy array to other formats"""

    @staticmethod
    def to_jsonl(npy_array, image_path, image_name, label_map: dict):
        """This method convert an Npy array to a jsonl format"""

        # loading image
        image = cv2.imread(image_path)

        # get the image width and height
        image_height, image_width, _ = image.shape

        # get the labels
        labels = npy_array

        # normalize the labels
        normalized_labels = []
        for label in labels:
            _, bounding_box, symbol = label
            symbol = str(symbol)
            normalized_labels.append({
                'label': label_map[symbol],
                'topX': bounding_box[0] / image_width,
                'topY': bounding_box[1] / image_height,
                'bottomX': bounding_box[2] / image_width,
                'bottomY': bounding_box[3] / image_height
            })

        # create the json line
        json_line = {
            'image_url': image_name,
            'image_details': {
                'format': 'jpg',
                'width': image_width,
                'height': image_height
            },
            'label': normalized_labels
        }

        return json_line
