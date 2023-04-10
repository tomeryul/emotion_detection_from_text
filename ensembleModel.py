#  !pip install tensorflow
#  !pip install transformers

import pickle
import tensorflow as tf
import transformers
# from transformers import TFBertModel
from transformers import AutoTokenizer


class TE_Model:

    def __init__(self):
        self.ML_model = self.load_ML_Model()
        self.DL_model = self.load_DL_Model()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.encoded_dict = {0: 'joy', 1: 'fear', 2: 'anger', 3: 'sadness', 4: 'disgust', 5: 'shame'}
        self.learned_dataSet_size = 7480
        self.emotionDict = {0: 1084, 1: 1078, 2: 1080, 3: 1079, 4: 1057, 5: 2102}

    def load_ML_Model(self):
        with open('./machine_learning_model', 'rb') as handle:
            model = pickle.load(handle)
            return model

    def load_DL_Model(self):
        loadedModel = tf.keras.models.load_model('./deeplearning_model.h5',
                                                 custom_objects={"TFBertModel": transformers.TFBertModel})
        return loadedModel

    def text_to_vector(self, text):
        x_val = self.tokenizer(
            text=text,
            add_special_tokens=True,
            max_length=70,
            truncation=True,
            padding='max_length',
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=True,
            verbose=True)

        return x_val

    def DL_Model_Results(self, text):
        x_val = self.text_to_vector(text)
        the_prediction = self.DL_model.predict(
            {'input_ids': x_val['input_ids'], 'attention_mask': x_val['attention_mask']}) * 100

        return list(the_prediction[0])

    def ML_Model_Results(self, text):
        start_arr = [1, 1, 1, 1, 1, 1]
        for word in text.split():
            if word in self.ML_model:
                start_arr = [x * y for x, y in zip(start_arr, self.ML_model[word])]
            else:
                start_arr = [x * y for x, y in
                             zip(start_arr, [1 / self.learned_dataSet_size, 1 / self.learned_dataSet_size,
                                             1 / self.learned_dataSet_size, 1 / self.learned_dataSet_size,
                                             1 / self.learned_dataSet_size, 1 / self.learned_dataSet_size])]

        help_arr = [self.emotionDict[0] / self.learned_dataSet_size, self.emotionDict[1] / self.learned_dataSet_size,
                    self.emotionDict[2] / self.learned_dataSet_size, self.emotionDict[3] / self.learned_dataSet_size,
                    self.emotionDict[4] / self.learned_dataSet_size, self.emotionDict[5] / self.learned_dataSet_size]
        the_prediction = [(x * y) for x, y in zip(help_arr, start_arr)]

        the_prediction = self.normalize_the_prediction(the_prediction)

        return the_prediction

    def normalize_the_prediction(self, the_prediction):
        normalized_arr = [0, 0, 0, 0, 0, 0]
        the_inset = 1
        for i in range(6):
            low = -1
            the_index = 0
            for index, num in enumerate(the_prediction):
                if num >= low:
                    low = num
                    the_index = index

            normalized_arr.pop(the_index)
            normalized_arr.insert(the_index, the_inset)
            the_prediction[the_index] = -1
            the_inset -= 0.075

        return normalized_arr

    def res_of_two_models(self, text):
        text = self.preper_text(text)

        ML_prediction = self.ML_Model_Results(text)
        DL_prediction = self.DL_Model_Results(text)

        # print("machine learning model: ", ML_prediction)
        # print("deap learning model: ", DL_prediction)

        ensemble_models = [(x*y) for x, y in zip(ML_prediction, DL_prediction)]

        return ensemble_models

    def preper_text(self, text):
        text = text.lower()
        text = "".join([letter if 97 <= ord(letter) <= 122 else " " for letter in text])

        return text

    def better_printing(self, answer):

        print()
        for i in range(len(answer)):
            print(self.encoded_dict[i], ":", answer[i])
        print()

