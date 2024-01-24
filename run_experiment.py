from main_sd import stable_glass_sd
import pandas as pd
from box import Box
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import open_clip
import argparse
import tqdm
from PIL import Image
import spacy
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoConfig
import seaborn as sns

class Noun:
    def __init__(self, name):
        self.name = name
        self.adjectives = []

    def add_adjective(self, adjective):
        self.adjectives.append(adjective)

class Verb:
    def __init__(self, name):
        self.name = name
        self.subject = None

    def add_subject(self, subject):
        self.subject = subject

class ClipResult:
    def __init__(self, description, sentence, score, layer, index):
        self.description = description
        self.sentence = sentence
        self.score = score
        self.layer = layer
        self.index = index


def is_vowel(s):
    res = False
    if s.startswith('e') or s.startswith('o') or s.startswith('a') or s.startswith('i') or s.startswith('u'):
        res = True
    return res

def string_to_int(s):
    res = 0
    if s == 'one':
        res = 1
    elif s == 'two':
        res = 2
    elif s == 'three':
        res = 3
    elif s == 'four':
        res = 4
    else:
        raise ValueError(f'Unknown number: {s}')
    return res

class CompositionalItem:
    def __init__(self, set_type, params):
        self.set_type = set_type
        if not set_type:
            self.full_sentence = params.prompt
            self.order = params.order

        elif set_type == 'animal':
            self.animal = params.animal
            self.order = params.order
            a_or_an = 'an' if is_vowel(params.animal) else 'a'
            self.full_sentence = f"A photo of {a_or_an} {params.animal}."

        elif self.set_type == 'animal_object':
            self.animal = params.animal
            self.object = params.object
            self.order = params.order
            a_or_an_animal = 'an' if is_vowel(params.animal) else 'a'
            a_or_an_object = 'an' if is_vowel(params.object) else 'a'
            if self.order == 'animal_object':
                self.full_sentence = f"A photo of {a_or_an_animal} {params.animal} wearing {a_or_an_object} {params.object}."
            elif self.order == 'object_animal':
                self.full_sentence = f"A photo of {a_or_an_object} {params.object} wore by {a_or_an_animal} {params.animal}."
            else:
                raise ValueError(f'Unknown order: {self.order}')
        elif set_type == 'object_size':
            self.small_object = params.small_object
            self.big_object = params.big_object
            self.order = params.order
            print("small_object: ", params.small_object)
            print("big_object: ", params.big_object)
            a_or_an_big_object = 'an' if is_vowel(params.big_object) else 'a'
            a_or_an_small_object = 'an' if is_vowel(params.small_object) else 'a'

            if self.order == 'small_big':
                self.full_sentence = f"A photo of {a_or_an_small_object} {params.small_object} and {a_or_an_big_object} {params.big_object}."
            elif self.order == 'big_small':
                self.full_sentence = f"A photo of {a_or_an_big_object} {params.big_object} and {a_or_an_small_object} {params.small_object}."
        elif set_type == 'animal_popularity':
            self.popular_animal = params.popular_animal
            self.unpopular_animal = params.unpopular_animal
            self.order = params.order
            a_or_an_popular_animal = 'an' if is_vowel(params.popular_animal) else 'a'
            a_or_an_unpopular_animal = 'an' if is_vowel(params.unpopular_animal) else 'a'
            if self.order == 'popular_unpopular':
                self.full_sentence = (f"A photo of {a_or_an_popular_animal} {params.popular_animal} and "
                                      f"{a_or_an_unpopular_animal} {params.unpopular_animal}.")
            elif self.order == 'unpopular_popular':
                self.full_sentence = (f"A photo of {a_or_an_unpopular_animal} {params.unpopular_animal} and "
                                      f"{a_or_an_popular_animal} {params.popular_animal}.")
            else:
                raise ValueError(f'Unknown order: {self.order}')
        elif set_type == 'natural':
            self.base_object = params.base_object
            self.natural_object = params.natural_object
            self.unnatural_object = params.unnatural_object
            self.order = params.order

            a_or_an_base_object = 'an' if is_vowel(params.base_object) else 'a'
            a_or_an_natural_object = 'an' if is_vowel(params.natural_object) else 'a'
            a_or_an_unnatural_object = 'an' if is_vowel(params.unnatural_object) else 'a'

            if self.order == 'base_natural':
                self.full_sentence = f"A photo of {a_or_an_base_object} {params.base_object} and {a_or_an_natural_object} {params.natural_object}."
            elif self.order == 'base_unnatural':
                self.full_sentence = f"A photo of {a_or_an_base_object} {params.base_object} and {a_or_an_unnatural_object} {params.unnatural_object}."

        elif set_type == 'animal_acts':
            self.order = params.order
            self.color1 = params.color1
            self.animal = params.animal
            self.color2 = params.color2
            self.object = params.object
            self.act = params.act
            
            a_or_an_color_1 = 'an' if is_vowel(params.color1) else 'a'
            a_or_an_color_2 = 'an' if is_vowel(params.color2) else 'a'
            a_or_an_act = 'an' if is_vowel(params.act) else 'a'

            if self.order == 'animal_act_object':
                self.full_sentence = f"A photo of {a_or_an_color_1} {params.color1} {params.animal} with {params.color2} {params.object} that is {params.act}."
            elif self.order == 'act_animal_object':
                self.full_sentence = f"A photo of {a_or_an_act} {params.act} {params.color1} {params.animal} with {params.color2} {params.object}."
            elif self.order == 'object_animal_act':
                self.full_sentence = f"A photo of {a_or_an_color_1} {params.color1} {params.object} with {params.color2} {params.animal} that is {params.act}."
            elif self.order == 'object_act_animal':
                self.full_sentence = f"A photo of {a_or_an_color_1} {params.color1} {params.object} on {a_or_an_act} {params.act} {params.color2} {params.animal}."
            else:
                raise ValueError(f'Unknown order: {self.order}')

        elif set_type == 'woman_wearing':
            self.order = params.order
            self.color1 = params.color1
            self.place = params.place
            self.color2 = params.color2
            self.big_object = params.big_object
            self.small_object = params.small_object
            
            a_or_an_place = 'an' if is_vowel(params.place) else 'a'
            a_or_an_big_object = 'an' if is_vowel(params.big_object) else 'a'
            a_or_an_small_object = 'an' if is_vowel(params.small_object) else 'a'
            a_or_an_color_1 = 'an' if is_vowel(params.color1) else 'a'
            a_or_an_color_2 = 'an' if is_vowel(params.color2) else 'a'

            if self.order == 'woman_place_big_small':
                self.full_sentence = (f"A photo of a woman standing in {a_or_an_place} {params.place} "
                                      f"wearing {a_or_an_color_1} {params.color1} {params.big_object} "
                                      f"and {a_or_an_color_2} {params.color2} {params.small_object}.")
            elif self.order == 'place_woman_big_small':
                self.full_sentence = (f"A photo of {a_or_an_place} {params.place}, where a woman is standing,"
                                    f"wearing {a_or_an_color_1} {params.color1} {params.big_object} "
                                      f"and {a_or_an_color_2} {params.color2} {params.small_object}.")
            elif self.order == 'woman_place_small_big':
                self.full_sentence = (f"A photo of a woman standing in {a_or_an_place} {params.place} "
                                        f"wearing {a_or_an_color_2} {params.color2} {params.small_object} "
                                        f"and {a_or_an_color_1} {params.color1} {params.big_object}.")
            elif self.order == 'small_first':
                self.full_sentence = (f"A photo of {a_or_an_color_2} {params.color2} {params.small_object} "
                                        f"and {a_or_an_color_1} {params.color1} {params.big_object} "
                                      f"worn by a woman, standing in {a_or_an_place} {params.place}.")
            elif self.order == 'big_first':
                self.full_sentence = (f"A photo of {a_or_an_color_1} {params.color1} {params.big_object} "
                                        f"and {a_or_an_color_2} {params.color2} {params.small_object} "
                                      f"worn by a woman, standing in {a_or_an_place} {params.place}.")

            else:
                raise ValueError(f'Unknown order: {self.order}')

        elif set_type == 'shapes':
            self.order = params.order
            self.color1 = params.color1
            self.color2 = params.color2
            self.color3 = params.color3
            self.shape1 = params.shape1
            self.shape2 = params.shape2
            self.surface = params.surface

            a_or_an_color_1 = 'an' if is_vowel(params.color1) else 'a'
            a_or_an_color_2 = 'an' if is_vowel(params.color2) else 'a'
            a_or_an_color_3 = 'an' if is_vowel(params.color3) else 'a'

            if self.order == 'shapes_surface':
                self.full_sentence = (f"{a_or_an_color_1} {params.color1} {params.shape1} and {a_or_an_color_2} {params.color2} {params.shape2}"
                                      f" on {a_or_an_color_3} {params.color3} {params.surface}.")
            elif self.order == 'surface_shapes':
                self.full_sentence = (f"{a_or_an_color_3} {params.color3} {params.surface} with {a_or_an_color_1} {params.color1} {params.shape1} "
                                      f"and {a_or_an_color_2} {params.color2} {params.shape2}.")

        elif set_type == 'celebs':
            self.order = params.order
            self.celeb = params.celeb
            self.full_sentence = f"A photo of {params.celeb}."

        elif set_type == 'things':
            self.order = params.order
            self.thing = params.thing

            a_or_an_thing = 'an' if is_vowel(params.thing) else 'a'
            self.full_sentence = f"A photo of {a_or_an_thing} {params.thing}."

        elif set_type == 'thing_color':
            self.order = params.order
            self.thing = params.thing
            self.color = params.color

            a_or_an_color = 'an' if is_vowel(params.color) else 'a'
            self.full_sentence = f"A photo of {a_or_an_color} {params.color} {params.thing}."

        elif set_type == 'two_things_color':
            self.order = params.order
            self.thing1 = params.thing1
            self.thing2 = params.thing2
            self.color1 = params.color1
            self.color2 = params.color2


            a_or_an_color_1 = 'an' if is_vowel(params.color1) else 'a'
            a_or_an_color_2 = 'an' if is_vowel(params.color2) else 'a'
            self.full_sentence = (f"A photo of {a_or_an_color_1} {params.color1} {params.thing1} and "
                                  f"{a_or_an_color_2} {params.color2} {params.thing2}.")


        elif set_type == 'gender_bias':
            self.order = params.order
            self.sentence = params.sentence
            self.unbiased_sentence = params.unbiased_sentence
            self.biased_sentence = params.biased_sentence
            a_or_an_sentence = 'an' if is_vowel(params.sentence) else 'a'
            if params.variation == 'biased':
                self.full_sentence = f"A photo of {a_or_an_sentence} {params.biased_sentence}."
            else:
                self.full_sentence = f"A photo of {a_or_an_sentence} {params.sentence}."
        elif set_type == 'general_bias':
            self.order = params.order
            self.sentence = params.sentence
            self.unbiased_sentence = params.unbiased_sentence
            self.biased_sentence = params.biased_sentence
            a_or_an_sentence = 'an' if is_vowel(params.sentence) else 'a'
            self.full_sentence = f"A photo of {a_or_an_sentence} {params.sentence}."
        elif set_type == 'relations':
            self.order = params.order
            self.first_thing = params.first_thing
            self.second_thing = params.second_thing
            self.relation = params.relation

            a_or_an_first_thing = 'an' if is_vowel(params.first_thing) else 'a'
            a_or_an_second_thing = 'an' if is_vowel(params.second_thing) else 'a'

            self.full_sentence = (f"A photo of {a_or_an_first_thing} {params.first_thing} {params.relation} "
                                  f"{a_or_an_second_thing} {params.second_thing}.")

        elif set_type == 'counting':
            self.order = params.order
            self.number_of_objects = params.number_of_objects
            self.object = params.object
            plural_ending = 's' if string_to_int(self.number_of_objects) > 1 else ''
            self.full_sentence = f"A photo of {params.number_of_objects} {params.object}{plural_ending}."

        else:
            raise ValueError(f'Unknown set type: {set_type}')

    def get_sentences_dict(self):
        # TODO add here the sentence dict for each set type
        if self.set_type == 'animal':
            # sentences_dict = {
            #     'animal': self.animal,
            # }
            sentences_dict = {
                'GoodQuality': f"The photo is of a good quality",
                "IsAnimal": f"The photo is of an animal",
                "IsMammal": f"The photo is of a mammal",
                "SameFur": f"The animal in the photo has the same fur as {self.animal}",
                "SameFace": f"The animal in the photo has the same face as {self.animal}",
            }
        elif self.set_type == 'animal_object':
            sentences_dict = {
                'animal': self.animal,
                'object': self.object,
                'animal with object': f"{self.animal} with {self.object}",

            }
        elif self.set_type == 'object_size':
            sentences_dict = {
                'big object': self.big_object,
                'small object': self.small_object,
                'big and small object': f"{self.big_object} and {self.small_object}",
            }
        elif self.set_type == 'animal_popularity':
            sentences_dict = {
                'popular animal': self.popular_animal,
                'unpopular animal': self.unpopular_animal,
                'popular and unpopular animal': f"{self.popular_animal} and {self.unpopular_animal}",
            }
        elif self.set_type == 'natural':
            sentences_dict = {
                'base_object': self.base_object,
                'natural object': self.natural_object,
                'unnatural object': self.unnatural_object,
                # 'natural_and_unnatural_object': f"{self.natural_object} and {self.unnatural_object}",
                'base and natural object': f"{self.base_object} and {self.natural_object}",
                'base and unnatural object': f"{self.base_object} and {self.unnatural_object}",
            }
        elif self.set_type == 'animal_acts':
            sentences_dict = {
                'animal': self.animal,
                'animal with object': f"{self.animal} with {self.object}",
                'animal act': f"{self.animal} {self.act}",
                'animal color': f"{self.color1} {self.animal}",
                'object color': f"{self.color2} {self.object}",
                'object': self.object,
                'act': self.act,
        }
        elif self.set_type == 'woman_wearing':
            sentences_dict = {
                # 'full_sentence': compositional_item.full_sentence,
                'place': self.place,
                'big object': self.big_object,
                'small object': self.small_object,
                'woman in place': f'woman in {self.place}',
                'woman': 'woman',
            }

        elif self.set_type == 'shapes':
            sentences_dict = {
                # 'full_sentence': compositional_item.full_sentence,
                'first shape': self.shape1,
                'second shape': self.shape2,
                'surface': self.surface,
                'shapes': f'{self.shape1} and {self.shape2}',
                'shapes on surface': f'{self.shape1} and {self.shape2} on {self.surface}',
            }
        elif self.set_type == 'celebs':
            sentences_dict = {
                'celeb': self.celeb,
            }
        elif self.set_type == 'things':
            sentences_dict = {
                'object': self.thing,
            }
        elif self.set_type == 'thing_color':
            sentences_dict = {
                'object': self.thing,
                'object with color': f'{self.color} {self.thing}',
                'color': self.color,
            }
        elif self.set_type == 'two_things_color':
            sentences_dict = {
                'first object': self.thing1,
                'first_color': f'{self.color1}',
                # 'first_color_thing': f'{self.color1} {self.thing1}',
                'second object': self.thing2,
                # 'correct colors': f'{self.color1} {self.thing1} and {self.color2} {self.thing2}',
                # 'wrong colors': f'{self.color2} {self.thing1} and {self.color1} {self.thing2}',
                'second_color': f'{self.color2}',
                # 'second_color_thing': f'{self.color2} {self.thing2}',
                # 'first_and_second_thing': f'{self.thing1} and {self.thing2}',
                # 'full': f'{self.color1} {self.thing1} and {self.color2} {self.thing2}',
            }
        elif self.set_type == 'gender_bias' or self.set_type == 'general_bias':
            biased = None
            if 'male' in self.sentence and 'female' not in self.sentence:
                biased = self.sentence.replace('male', 'female')
            elif 'female' in self.sentence:
                biased = self.sentence.replace('female', 'male')
            sentences_dict = {
                'anti_biased': self.sentence,
                'neutral': self.unbiased_sentence,
            }
            if biased is not None:
                sentences_dict['biased'] = biased
        elif self.set_type == 'relations':
            sentences_dict = {
                'right relation': f'{self.first_thing} {self.relation} {self.second_thing}',
                'first thing': self.first_thing,
                'second thing': self.second_thing,
                'opposite relation': f'{self.second_thing} {self.relation} {self.first_thing}',
            }
        elif self.set_type == 'counting':
            sentences_dict = {
                'one object': f'one {self.object}',
                'two objects (correct)': f'two {self.object}s',
                'three objects': f'three {self.object}s',
                'four objects': f'four {self.object}s',
                'five objects': f'five {self.object}s',
            }
        else:
            raise ValueError(f'Unknown set type: {self.set_type}')
        return sentences_dict

    def set_pos(self, pos):
        self.nouns = pos['nouns']
        self.verbs = pos['verbs']
        self.root = pos['root']

class CompositionalExperiment:
    def __init__(self, main_folder_name='diffusion_outputs', img_num=4, blip_model_size='xl'):
        self.dataset = None
        self.main_folder_name = main_folder_name
        self.img_num = img_num

        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')


        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        self.blip_model_name = f"Salesforce/blip2-flan-t5-{blip_model_size}"
        self.blip_processor = Blip2Processor.from_pretrained(self.blip_model_name)
        self.blip_config = AutoConfig.from_pretrained(self.blip_model_name)
        self.blip_config.max_new_tokens = 64
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            self.blip_model_name, torch_dtype=torch.float16, config=self.blip_config
        )

    def create_dataset(self, set_type, params):
        compositional_items = []
        if not set_type:
            if 'input_filename' not in params or not params.input_filename:
                if not params.prompt:
                    raise ValueError('Prompt must be provided if input folder is not provided')
                else:
                    current_item = CompositionalItem(set_type, Box({
                        'set_type': set_type,
                        'prompt': params.prompt,
                        'order': 'None'
                    }))
                    pos = self.get_pos(params.prompt)
                    current_item.set_pos(pos)
                    compositional_items.append(current_item)
            else:
                path_to_input = os.path.join(params.input_filename)
                if not os.path.exists(path_to_input):
                    path_to_input = os.path.join('inputs', params.input_filename)
                    if not os.path.exists(path_to_input):
                        raise ValueError(f'Input file does not exist: {path_to_input}')
                with open(path_to_input, 'r') as f:
                    prompts = f.readlines()
                if params.number_of_inputs != -1: # limit the number of inputs
                    prompts = prompts[:params.number_of_inputs]
                for prompt in prompts:
                    prompt = prompt.strip()
                    current_item = CompositionalItem(set_type, Box({
                        'set_type': set_type,
                        'prompt': prompt,
                        'order': 'None'
                    }))
                    pos = self.get_pos(prompt)
                    current_item.set_pos(pos)
                    compositional_items.append(current_item)
        elif set_type == 'animal_acts':
            for animal in params.animals:
                for object in params.objects:
                    for act in params.acts:
                        for color1 in params.colors:
                            for color2 in params.colors:
                                if color1 == color2:
                                    continue
                                for order in params.orders:
                                    compositional_items.append(CompositionalItem(set_type, Box({
                                        'set_type': set_type,
                                        'color1': color1,
                                        'animal': animal,
                                        'color2': color2,
                                        'object': object,
                                        'act': act,
                                        'order': order,
                                    })))

        elif set_type == 'woman_wearing':
            for place in params.places:
                for color1 in params.colors:
                    for color2 in params.colors:
                        if color1 == color2:
                            continue
                        for big_object in params.big_objects:
                            for small_object in params.small_objects:
                                for order in params.orders:
                                    compositional_items.append(CompositionalItem(set_type, Box({
                                        'set_type': set_type,
                                        'color1': color1,
                                        'place': place,
                                        'color2': color2,
                                        'big_object': big_object,
                                        'small_object': small_object,
                                        'order': order,
                                    })))
        elif set_type == 'shapes':
            for shape1 in params.shapes:
                for shape2 in params.shapes:
                    for color1 in params.colors:
                        for color2 in params.colors:
                            for color3 in params.colors:
                                for surface in params.surfaces:
                                    if (color1 == color2 or
                                            color2 == color3 or
                                            color1 == color3 or
                                            shape1 == shape2):
                                        continue
                                    for order in params.orders:
                                        compositional_items.append(CompositionalItem(set_type, Box({
                                            'set_type': set_type,
                                            'shape1': shape1,
                                            'shape2': shape2,
                                            'color1': color1,
                                            'color2': color2,
                                            'color3': color3,
                                            'surface': surface,
                                            'order': order,
                                        })))
        elif set_type == 'animal':
            compositional_items = [CompositionalItem(set_type, Box({
                'set_type': set_type,
                'animal': animal,
                'order': 'None'
            })) for animal in params.animals]
        elif set_type == 'animal_object':
            compositional_items = [CompositionalItem(set_type, Box({
                'set_type': set_type,
                'animal': animal,
                'object': curr_object,
                'order': order,
            })) for animal in params.animals
                                   for curr_object in params.objects
                                   for order in params.orders]
        elif set_type == 'object_size':
            for big_object in params.big_objects:
                for small_object in params.small_objects:
                    for order in params.orders:
                        print("big_object: ", big_object)
                        print("small_object: ", small_object)
                        print("order: ", order)
                        compositional_items.append(CompositionalItem(set_type, Box({
                            'set_type': set_type,
                            'big_object': big_object,
                            'small_object': small_object,
                            'order': order,
                        })))
        elif set_type == 'animal_popularity':
            compositional_items = [CompositionalItem(set_type, Box({
                'set_type': set_type,
                'popular_animal': popular_animal,
                'unpopular_animal': unpopular_animal,
                'order': order,
            })) for popular_animal in params.popular_animals
                                   for unpopular_animal in params.unpopular_animals
                                   for order in params.orders]

        elif set_type == 'natural':
            compositional_items = [CompositionalItem(set_type, Box({
                'set_type': set_type,
                'base_object': base_object,
                'natural_object': natural_object,
                'unnatural_object': unnatural_object,
                'order': order,
            })) for base_object in params.base_objects
                                   for natural_object in params.natural_objects
                                   for unnatural_object in params.unnatural_objects
                                   for order in params.orders]

        elif set_type == 'celebs':
            compositional_items = [CompositionalItem(set_type, Box({
                'set_type': set_type,
                'celeb': celeb,
                'order': 'None'
            })) for celeb in params.celebs]
        elif set_type == 'things':
            compositional_items = [CompositionalItem(set_type, Box({
                'set_type': set_type,
                'thing': thing,
                'order': 'None'
            })) for thing in params.things]
        elif set_type == 'gender_bias':
            compositional_items = [CompositionalItem(set_type, Box({
                'set_type': set_type,
                'sentence': sentence,
                'unbiased_sentence': unbiased_sentence,
                'biased_sentence': biased_sentence,
                'order': 'None',
                'variation': params.variation,
            })) for (sentence, unbiased_sentence, biased_sentence) in zip(params.sentences, params.unbiased_sentences,
                                                                            params.biased_sentences)]
        elif set_type == 'general_bias':
            compositional_items = [CompositionalItem(set_type, Box({
                'set_type': set_type,
                'sentence': sentence,
                'unbiased_sentence': unbiased_sentence,
                'order': 'None',
                'variation': params.variation,
            })) for (sentence, unbiased_sentence) in zip(params.sentences, params.unbiased_sentences)]
        elif set_type == 'relations':
            compositional_items = [CompositionalItem(set_type, Box({
                'set_type': set_type,
                'first_thing': first_thing,
                'second_thing': second_thing,
                'relation': relation,
                'order': 'None',
            })) for first_thing in params.objects
                                   for second_thing in params.objects
                                   for relation in params.relations if first_thing != second_thing]
        elif set_type == 'counting':
            compositional_items = [CompositionalItem(set_type, Box({
                'set_type': set_type,
                'number_of_objects': number_of_objects,
                'object': object_name,
                'order': 'None',
            })) for number_of_objects in params.number_of_objects
                                   for object_name in params.objects]
        elif set_type == 'thing_color':
            compositional_items = [CompositionalItem(set_type, Box({
                'set_type': set_type,
                'thing': thing,
                'color': color,
                'order': 'None'
            })) for thing in params.things
                                   for color in params.colors]
        elif set_type == 'two_things_color':
            for color1 in params.colors:
                for color2 in params.colors:
                    if color1 == color2:
                        continue
                    for thing1 in params.things:
                        for thing2 in params.things:
                            if thing1 == thing2:
                                continue
                            compositional_items.append(CompositionalItem(set_type, Box({
                                'set_type': set_type,
                                'thing1': thing1,
                                'color1': color1,
                                'thing2': thing2,
                                'color2': color2,
                                'order': 'None'
                            })))
        else:
            raise ValueError(f'Unknown set type: {set_type}')


        self.dataset = compositional_items
        return True

    def get_dataset(self):
        return self.dataset
    def run_experiment(self, set_type, params):
        self.create_dataset(set_type=set_type, params=params)
        compositional_items = self.get_dataset()

        if params.number_of_inputs != -1: # limit the number of inputs
            compositional_items = compositional_items[:params.number_of_inputs]

        print("Number of compositional items: ", len(compositional_items))

        # print("sentences: ", sentences)
        args_list = []
        for compositional_item in compositional_items:
            print("sentence: ", compositional_item.full_sentence)
            args = {
                'prompt': compositional_item.full_sentence,
                'lens_type': 'full',
                'part': 'encoder',
                'mlp': False,
                'word': False,
                'not_direct': False,
                'pad': False,
                'pre_prompt': '',
                'diffusion_encoder_input': None,
                'img_num': self.img_num,
                'model_key': params.model_key,
                'max_length': 10,
                'main_folder_name': self.main_folder_name,
                'input_filename': params.input_filename,
                'skip_all_layers': params.skip_all_layers,
                'start_layer': params.start_layer,
                'end_layer': params.end_layer,
                'step_layer': params.step_layer,
                'explain_other_model': params.explain_other_model,
                'per_token': params.per_token,
            }

            args = Box(args)
            args_list.append(args)
        stable_glass_sd(args_list)
        # if 'sd' in params.model_key:
        #     stable_glass_sd(args_list)
        # else:
        #     print("Using previous code for DF")
        #     from main import stable_glass
        #     stable_glass(args_list)



    def get_clip_scores(self, img_paths, sentences_dict, softmax, is_batch=False, full_sentence=None, order=None, use_blip=False):
        df_clip_scores = pd.DataFrame(columns=['sentence', 'object_type', 'object_name', 'score', 'order', 'layer', 'index'])
        sentences = list(sentences_dict.values())
        # scores = self.calculate_clip_score(img, sentences)

        scores = self.calculate_openclip_score(prompts=sentences, img_path=img_paths, softmax=softmax, is_list=is_batch,
                                                   is_blip=use_blip)

        print("Scores: ", scores)

        # scores_after_norm_by_simple_generation = 0
        # all_scores = []
        # clip_scores = []
        if not is_batch:
            img_paths = [img_paths]
        for image_idx, image_path in enumerate(img_paths):
            for sentence_key, sentence_val, score in (
                    zip(sentences_dict.keys(), sentences_dict.values(), scores[image_idx])):
                # print("Score: ", sentence_key, sentence_val, score)
                # print("Scores: ",scores)
                idx = image_idx % self.img_num
                layer = image_idx // self.img_num
                df_clip_scores = pd.concat([df_clip_scores,
                                            pd.DataFrame([[full_sentence, sentence_key, sentence_val,
                                                           score, order, layer, idx]],
                                                         columns=['sentence', 'object_type', 'object_name',
                                                                  'score', 'order', 'layer', 'index'])])


        return df_clip_scores

    def get_pos(self, full_sentence):
        noun_dict = {}
        verb_dict = {}
        spacy.load('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(full_sentence)
        root = None
        for token in doc:
            # print(f'text: {token.text}, pos: {token.pos_}, dep: {token.dep_}, '
            #       f'children: {token.children}, children text: {[child.text for child in token.children]}')
            if token.pos_ == 'NOUN':
                new_noun = Noun(token.text)
                for child in token.children:
                    if child.pos_ == 'ADJ':
                        new_noun.add_adjective(child.text)
                noun_dict[token.text] = new_noun
            if token.pos_ == 'VERB':
                new_verb = Verb(token.text)
                for child in token.children:
                    if child.pos_ == 'NOUN':
                        new_verb.add_subject(child.text)
                verb_dict[token.text] = new_verb
            if token.dep_ == 'ROOT':
                root = token.text

        return {
            'nouns': noun_dict,
            'verbs': verb_dict,
            'root': root,
        }
    def get_sentences_dict_from_pos(self, pos):
        nouns = pos['nouns']
        verbs = pos['verbs']
        root = pos['root']
        sentences_dict = {}
        if root is not None:
            sentences_dict[f"root"] = f"A photo of {root}"
        for noun_idx ,(noun_name, noun_object) in enumerate(nouns.items()):
            sentences_dict[f"object{noun_idx}"] = f"A photo of a {noun_object.name}"
            if len(noun_object.adjectives) > 0:
                sentences_dict[f"object{noun_idx}_adjectives"] = (f"A photo of {' '.join(noun_object.adjectives)} "
                                                                           f"{noun_object.name}")

        for verb_idx, (verb, verb_object) in enumerate(verbs.items()):
            if verb_object.subject != None:
                sentences_dict[f"verb{verb_idx}_subject"] = f"A photo of {verb_object.name} {verb_object.subject}"
            else:
                sentences_dict[f"verb{verb_idx}"] = f"A photo of {verb_object.name}"
        return sentences_dict

    def get_texts_for_clip_scores(self, compositional_item, test_type=None):
        if compositional_item.set_type is None:
            pos = self.get_pos(full_sentence=compositional_item.full_sentence)
            sentences_dict = self.get_sentences_dict_from_pos(pos=pos)
        else:
            sentences_dict = compositional_item.get_sentences_dict()
        return sentences_dict


    def plot_clip_scores(self, img, clip_scores, layer, index, path_to_plots, full_sentence,
                         do_save=False, do_show=False):
        if not os.path.exists(path_to_plots):
            os.makedirs(path_to_plots)

        text_path = os.path.join(path_to_plots, f"{full_sentence}_{layer}_{layer}.txt")
        with open(text_path, 'w') as f:
            for clip_score in clip_scores:
                f.write(f"{clip_score.description}: {clip_score.score}")
                f.write("\n")

        # show the image on a plot with the clip scores
        plt.imshow(img)
        plt.title(full_sentence + f"_{layer}_{index}")
        clip_scores_text = "\n".join([f"{clip_score.sentence}: {clip_score.score:2}" for clip_score in clip_scores])
        plt.xlabel(clip_scores_text)
        plt.tight_layout()
        if do_show:
            plt.show()
        if do_save:
            plt.savefig(os.path.join(path_to_plots, f"{full_sentence}_{layer}_{index}.png"))
            plt.savefig(os.path.join(path_to_plots, f"{full_sentence}_{layer}_{index}.pdf"), format='pdf', bbox_inches='tight')

        return
        
        

    # def plot_aggregated_plot_with_seaborn(self, df_object, path, model_key, sorted_by='Nothing', curr_object=None,
    #                                       is_normalized=False):
    #     is_normalized_ending = '_normalized' if is_normalized else ''
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     df_object = df_object.reset_index(drop=True)
    #
    #     # remove NaNs
    #     len_before_remove_nan = len(df_object)
    #     df_object = df_object.dropna()
    #     len_after_remove_nan = len(df_object)
    #     if len_before_remove_nan != len_after_remove_nan:
    #         print("The length of the df is different after remove NaNs")
    #         print("len_before_remove_nan: ", len_before_remove_nan)
    #         print("len_after_remove_nan: ", len_after_remove_nan)
    #     df_object = df_object.reset_index(drop=True)
    #
    #     avg_score = df_object['scores'].mean()
    #     std_score = np.std(df_object['scores'].to_numpy())
    #
    #     object_type = df_object.loc[0, sorted_by]
    #
    #     # plot
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     sns.lineplot(
    #         x=range(len(df_object['scores'])),
    #         y=df_object['scores'],
    #         hue=df_object[sorted_by],
    #         style=df_object[sorted_by],
    #         ax=ax,
    #     )
    #     # add error bands
    #     sns.lineplot(
    #         x=df_object['layer'],
    #         y=avg_score,
    #         linewidth=2,
    #         color="black",
    #         ax=ax,
    #     )
    #     sns.fill_between(
    #         x=df_object['layer'],
    #         y1=avg_score - std_score,
    #         y2=avg_score + std_score,
    #         color="black",
    #         alpha=0.2,
    #         ax=ax,
    #     )
    #
    #     # set title and labels
    #     ax.set_title(f'{sorted_by} - {curr_object} - average score')
    #     ax.set_xlabel('layer')
    #     ax.set_ylabel('score')
    #
    #     # add legend
    #     ax.legend()
    #
    #     # adjust layout
    #     plt.tight_layout()
    #
    #     # save to file
    #     plt.savefig(os.path.join(path, f'{model_key}_{sorted_by} - {curr_object}_with_std{is_normalized_ending}.png'))
    #     print("saved to: ",
    #           os.path.join(path, f'{model_key}_{sorted_by} - {curr_object}_with_std{is_normalized_ending}.png'))
    #     plt.show()

    def plot_aggregated_plot(self, df_object, path, model_key, sorted_by='Nothing', curr_object=None,
                             is_normalized=False):
        is_normalized_ending = '_normalized' if is_normalized else ''
        if not os.path.exists(path):
            os.makedirs(path)
        df_object = df_object.reset_index(drop=True)

        # remove NaNs
        len_before_remove_nan = len(df_object)
        df_object = df_object.dropna()
        len_after_remove_nan = len(df_object)
        if len_before_remove_nan != len_after_remove_nan:
            print("The length of the df is different after remove NaNs")
            print("len_before_remove_nan: ", len_before_remove_nan)
            print("len_after_remove_nan: ", len_after_remove_nan)
        df_object = df_object.reset_index(drop=True)

        avg_score = df_object['scores'].mean()
        std_score = np.std(df_object['scores'].to_numpy())

        object_type = df_object.loc[0, sorted_by]

        # plot
        plt.clf()
        plt.title(f'{sorted_by} - {curr_object} - average score')
        plt.xlabel('layer')
        plt.ylabel('score')
        plt.plot(avg_score, label=f'{object_type}')
        plt.fill_between(range(len(avg_score)), avg_score - std_score, avg_score + std_score, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        # save to file
        plt.savefig(os.path.join(path, f'{model_key}_{sorted_by} - {curr_object}_with_std{is_normalized_ending}.png'))
        plt.savefig(os.path.join(path, f'{model_key}_{sorted_by} - {curr_object}_with_std{is_normalized_ending}.pdf'), format='pdf', bbox_inches='tight')
        print("saved to: ", os.path.join(path, f'{model_key}_{sorted_by} - {curr_object}_with_std{is_normalized_ending}.png'))

        plt.show()

    def plot_aggregated_plot_all_in_one(self, df_results, column_name, path, model_key, is_normalized=False,
                                        curr_order=None):
        is_normalized_ending = '_normalized' if is_normalized else ''
        curr_order_ending = f'_{curr_order}' if curr_order is not None else ''
        plt.clf()
        plt.title(f'Group by {column_name} - average score {curr_order_ending} {is_normalized_ending}')
        plt.xlabel('layer')
        plt.ylabel('score')
        # plot each object type with different color
        print("Plotting: ", f"Group by {column_name} - average score")
        print("DF shape: ", df_results.shape)
        objects = df_results[column_name].unique()
        if len(objects) == 0:
            print("No unique objects")
            return
        else:
            print("Unique objects: ", objects)


        for curr_object in objects:
            print("object: ", curr_object)
            df_object = df_results[df_results[column_name] == curr_object]
            if len(df_object) == 0:
                print("The length of the df is 0 for object: ", curr_object)
                continue
            else:
                print("The length of the df is: ", len(df_object))

            df_object = df_object.reset_index(drop=True)
            # remove NaNs
            len_before_remove_nan = len(df_object)
            df_object = df_object.dropna()
            len_after_remove_nan = len(df_object)
            if len_before_remove_nan != len_after_remove_nan:
                print("The length of the df is different after remove NaNs")
                print("len_before_remove_nan: ", len_before_remove_nan)
                print("len_after_remove_nan: ", len_after_remove_nan)
            df_object = df_object.reset_index(drop=True)
            print("DF scores:", df_object['scores'])
            print("DF scores mean:", df_object['scores'].mean())

            avg_score = df_object['scores'].mean()
            std_score = np.std(df_object['scores'].to_numpy())
            object_group = df_object.loc[0, column_name]

            plt.plot(avg_score, label=f'{object_group}')
            plt.fill_between(range(len(avg_score)), avg_score - std_score, avg_score + std_score, alpha=0.2,
                                label=f'{object_group}')

        plt.legend(loc='best')
        plt.tight_layout()
        # save to file
        plt.savefig(os.path.join(path, f'Summary_{model_key}_{column_name}_with_sdt{is_normalized_ending}.png'))
        plt.savefig(os.path.join(path, f'Summary_{model_key}_{column_name}_with_sdt{is_normalized_ending}.pdf'), format='pdf', bbox_inches='tight')
        print("saved to: ", os.path.join(path, f'Summary_{model_key}_{column_name}_with_sdt{is_normalized_ending}.png'))
        print("saved to: ", os.path.join(path, f'Summary_{model_key}_{column_name}_with_sdt{is_normalized_ending}.pdf'))
        plt.show()

        plt.clf()
        plt.title(f'Group by {column_name} - average score')
        plt.xlabel('layer')
        plt.ylabel('score')
        # plot each object type with different color
        objects = df_results[column_name].unique()
        if len(objects) == 0:
            return
        for object in objects:
            df_object = df_results[df_results[column_name] == object]
            if len(df_object) == 0:
                continue
            df_object = df_object.reset_index(drop=True)

            # remove NaNs
            len_before_remove_nan = len(df_object)
            df_object = df_object.dropna()
            len_after_remove_nan = len(df_object)
            if len_before_remove_nan != len_after_remove_nan:
                print("The length of the df is different after remove NaNs")
                print("len_before_remove_nan: ", len_before_remove_nan)
                print("len_after_remove_nan: ", len_after_remove_nan)

            df_object = df_object.reset_index(drop=True)

            avg_score = df_object['scores'].mean()
            std_score = np.std(df_object['scores'].to_numpy())
            object_group = df_object.loc[0, column_name]

            plt.plot(avg_score, label=f'{object_group}')
            # plt.fill_between(range(len(avg_score)), avg_score - std_score, avg_score + std_score, alpha=0.2,
            #                  label=f'{object_group}')

        plt.legend(loc='best')
        plt.tight_layout()
        current_endind = f'_{curr_order}' if curr_order is not None else ''
        # save to file
        plt.savefig(os.path.join(path, f'Summary_{model_key}{current_endind}_{column_name}_without_std{is_normalized_ending}.png'),  format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(path, f'Summary_{model_key}{current_endind}_{column_name}_without_std{is_normalized_ending}.pdf'), format='pdf', bbox_inches='tight')
        plt.show()

    def create_aggregation_plot(self, df_results, path, args, is_normalized=False):
        """
        columns=['sentence', 'object_type', 'object_name', 'scores']))
        """
        if not os.path.exists(path):
            os.makedirs(path)
        unique_object_types = df_results['object_type'].unique()
        if len(unique_object_types) > 1:
            for curr_object in unique_object_types:
                df_object = df_results[df_results['object_type'] == curr_object]
                self.plot_aggregated_plot(df_object=df_object, curr_object=curr_object,
                                          path=path, sorted_by='object_type',
                                          model_key=args.model_key, is_normalized=is_normalized)


        # plot all the object types on the same plot
        self.plot_aggregated_plot_all_in_one(df_results=df_results, column_name='object_type', path=path,
                                             model_key=args.model_key, is_normalized=is_normalized)

        split_by_order_path = os.path.join(path, 'split_by_order')
        print("path to split: ", split_by_order_path)
        if not os.path.exists(split_by_order_path):
            os.makedirs(split_by_order_path)
            print("Created path: ", split_by_order_path)

        print("unique object types: ", df_results['order'].unique())
        for curr_order in df_results['order'].unique():
            print("curr_order: ", curr_order)
            df_object_order = df_results[df_results['order'] == curr_order]
            if len(df_object_order) == 0:
                print("The length of the df is 0 for object: ", curr_order)
                continue
            else:
                print("The length of the df is: ", len(df_object_order))
            print("Plotting aggregated plot for order: ", curr_order)
            self.plot_aggregated_plot(df_object=df_object_order, curr_object=curr_order,
                                      path=split_by_order_path, sorted_by='order', model_key=args.model_key,
                                      is_normalized=is_normalized)
            print("After plot_aggregated_plot")
            unique_object_order_types = df_object_order['object_type'].unique()
            if len(unique_object_order_types) > 1:
                print("unique_object_order_types: ", unique_object_order_types)
                for curr_object in unique_object_order_types:
                    df_unique_object_order = df_object_order[df_object_order['object_type'] == curr_object]
                    self.plot_aggregated_plot(df_object=df_unique_object_order, curr_object=curr_object,
                                              path=split_by_order_path, sorted_by='object_type', model_key=args.model_key,
                                              is_normalized=is_normalized)
            else:
                print("No unique object order types")
            df_results_for_order = df_results[df_results['order'] == curr_order]
            print("Per order df size: ", df_results_for_order.shape)
            self.plot_aggregated_plot_all_in_one(df_results=df_results_for_order, column_name='object_type',
                                                 model_key=args.model_key,
                                                    path=split_by_order_path, curr_order=curr_order)

            # for curr_object in df_results['object_name'].unique():
            #     df_object = df_results[df_results['object_name'] == curr_object]
            #     self.plot_aggregated_plot(df_object=df_object, curr_object=curr_object,
            #                               path=path, sorted_by='object_name', model_key=args.model_key,
            #                               is_normalized=is_normalized)

        print("Done with order")
        # plot all the object types on the same plot
        # self.plot_aggregated_plot_all_in_one(df_results=df_results, column_name='object_name', path=path)

        return True


    def preprocess_descriptions(self, descriptions, threshold, model_key, set_type=None, only_avg=True):
        if not only_avg:
            if set_type == 'animal_acts':
                # remove all items with 'random' in the key
                # descriptions = {k: v for k, v in descriptions.items() if 'random' not in k}
                num_of_layers = 13 if model_key == 'sd1.4' else 25
                scores = descriptions['animal_with_object'][1]
                for layer in range(num_of_layers):
                    for index in range(4):
                        descriptions['animal'][1][index, layer] += scores[index, layer]
                        descriptions['object'][1][index, layer] += scores[index, layer]
                del descriptions['animal_with_object']
                scores = descriptions['animal_act'][1]
                for layer in range(num_of_layers):
                    for index in range(4):
                        descriptions['animal'][1][index, layer] += scores[index, layer]
                        descriptions['act'][1][index, layer] += scores[index, layer]
                del descriptions['animal_act']
            elif set_type == 'woman_wearing':
                raise NotImplementedError
            elif set_type == 'shapes':
                raise NotImplementedError
            else: # set_type is unknown or None
                print("set_type is unknown or None")
                raise NotImplementedError
            # general preprocessing
            for description_key, description_value in descriptions.items():
                for layer in range(25):
                    for index in range(4):
                        if descriptions[description_key][1][index, layer] > threshold:
                            descriptions[description_key][1][index, layer] = 1
                        else:
                            descriptions[description_key][1][index, layer] = 0
        normalized_descriptions = {}
        for description_key, description_value in descriptions.items():
            description_value_score = np.mean(descriptions[description_key]['scores'], axis=0)
            descriptions[description_key]['scores'] = description_value_score
            if np.max(description_value_score) - np.min(description_value_score) > 0:
                normalized_scores = ((description_value_score - np.min(description_value_score)) /
                                     (np.max(description_value_score) - np.min(description_value_score)))
            else:
                print("Min and max are the same")
                normalized_scores = description_value_score

            normalized_descriptions[description_key] = {
                'sentence': description_value['sentence'],
                'scores': normalized_scores,
                'object_type': description_value['object_type'],
                'object_name': description_value['object_name']
            }

        return {
            'normalized_descriptions': normalized_descriptions,
            'descriptions': descriptions,
        }

    # def plot_scores_with_seaborn(self, descriptions, normalized_descriptions, path, create_plot_per_sentence):
    #     """Plots the scores for each description and normalized description.
    #
    #     Args:
    #         descriptions: A dictionary of descriptions to scores.
    #         normalized_descriptions: A dictionary of normalized descriptions to scores.
    #         path: The path to save the plots to.
    #         create_plot_per_sentence: Whether to create a plot for each sentence.
    #     """
    #
    #     if create_plot_per_sentence:
    #         for description, scores in descriptions.items():
    #             print("description: ", description)
    #             # plot the scores
    #             fig, ax = plt.subplots(figsize=(10, 6))
    #             sns.lineplot(
    #                 x="layer",
    #                 y="scores",
    #                 data=scores,
    #                 ax=ax,
    #                 label=f"{description}",
    #             )
    #
    #             # set title and labels
    #             ax.set_title(f"{scores['sentence']}, {scores['object_type']}, {scores['object_name']}")
    #             ax.set_xlabel("layer")
    #             ax.set_ylabel("score")
    #
    #             # add legend
    #             ax.legend()
    #
    #             # adjust layout
    #             plt.tight_layout()
    #
    #             # save to file
    #             plt.savefig(os.path.join(path, f"{scores['sentence']}_{description}.png"))
    #             print("Plot saved successfully in path: ",
    #                   os.path.join(path, f"{scores['sentence']}_{description}.png"))
    #             plt.show()
    #
    #     for normalized_description, scores in normalized_descriptions.items():
    #         print("normalized_description: ", normalized_description)
    #         # plot the scores
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         sns.lineplot(
    #             x="layer",
    #             y="scores",
    #             data=scores,
    #             ax=ax,
    #             label=f"{normalized_description}",
    #         )
    #
    #         # set title and labels
    #         ax.set_title(f"Normalized {scores['sentence']}, {scores['object_type']}, {scores['object_name']}")
    #         ax.set_xlabel("layer")
    #         ax.set_ylabel("score")
    #
    #         # add legend
    #         ax.legend()
    #
    #         # adjust layout
    #         plt.tight_layout()
    #
    #         # save to file
    #         plt.savefig(os.path.join(path, f"{scores['sentence']}_{normalized_description}_Norm.png"))
    #         print("Plot saved successfully in path: ",
    #               os.path.join(path, f"{scores['sentence']}_{normalized_description}_Norm.png"))
    #         plt.show()

    def create_plot(self,model_key, df_results, full_sentence, path,
                    set_type=None, create_plot_per_sentence=False):
        threshold = 0.3
        descriptions = {}
        df_sentence = df_results[df_results['sentence'] == full_sentence]
        df_sentence = df_sentence.sort_values(by=['layer', 'index'])
        df_sentence = df_sentence.reset_index(drop=True)
        # iterate over the sentences and create score history (for all layer) per description
        for index, row in df_sentence.iterrows():
            description = row['object_name']
            sentence = row['sentence']
            score = row['score']
            if description not in descriptions.keys():
                max_layer = df_results['layer'].max() + 1
                descriptions[description] = {
                    'sentence': sentence,
                    'scores': np.zeros((4, max_layer)),
                    'object_type': row['object_type'],
                    'object_name': row['object_name']
                }

            descriptions[description]['scores'][row['index'], row['layer']] = score.item()
        preprocess_descriptions = self.preprocess_descriptions(descriptions, threshold, model_key=model_key,
                                                               set_type=set_type, only_avg=True)
        normalized_descriptions = preprocess_descriptions['normalized_descriptions']
        descriptions = preprocess_descriptions['descriptions']

        # print("descriptions.scores: ", descriptions['scores'])
        # print("normalized_descriptions.scores: ", normalized_descriptions['scores'])

        if not os.path.exists(path):
            os.makedirs(path)

        # self.plot_scores_with_seaborn(descriptions=descriptions, normalized_descriptions=normalized_descriptions, path=path,
        #                          create_plot_per_sentence=create_plot_per_sentence)
        # plot the scores
        if create_plot_per_sentence:
            for description, scores in descriptions.items():
                print("description: ", description)
                # plot the scores
                plt.clf()
                plt.title(f'{scores["sentence"]}, {scores["object_type"]}, {scores["object_name"]}')
                plt.xlabel('layer')
                plt.ylabel('score')
                plt.plot(scores["scores"], label=f'{description}')

                plt.legend()
                plt.tight_layout()
                # save to file
                plt.savefig(os.path.join(path, f'{full_sentence}_{description}.png'))
                plt.savefig(os.path.join(path, f'{full_sentence}_{description}.pdf'), format='pdf', bbox_inches='tight')
                print("Plot saved successfully in path: ", os.path.join(path, f'{full_sentence}_{description}.png'))
                plt.show()

            for normalized_description, scores in  normalized_descriptions.items():
                print("normalized_description: ", normalized_description)
                # plot the scores
                plt.clf()
                plt.title(f'Normalized {scores["sentence"]}, {scores["object_type"]}, {scores["object_name"]}')
                plt.xlabel('layer')
                plt.ylabel('score')
                plt.plot(scores['scores'], label=f'{normalized_description}')
                # Y axis should be from 0 to 1
                plt.ylim(0, 1)
                plt.legend()
                plt.tight_layout()
                # save to file
                plt.savefig(os.path.join(path, f'{full_sentence}_{normalized_description}_Norm.png'))
                plt.savefig(os.path.join(path, f'{full_sentence}_{normalized_description}_Norm.pdf'), format='pdf', bbox_inches='tight')
                print("Plot saved successfully in path: ", os.path.join(path, f'{full_sentence}_{normalized_description}.png'))
                plt.show()

        return preprocess_descriptions


    def evaluate_results(self, set_type, params, iterate_by='folder', plot_clip=False,
                         create_plot_per_sentence=False, evaluate_from_scratch=False):
        # check if the results already exist
        blip_ending = f'_blip_{args.blip_model_size}' if params.use_blip else ''
        path_full = os.path.join(args.folder_name, f'df_results_{params.model_key}_full{blip_ending}.pkl')
        path_normalized = os.path.join(args.folder_name, f'normalized_results_{params.model_key}_per_object{blip_ending}.pkl')
        path_per_object = os.path.join(args.folder_name, f'df_results_{params.model_key}_per_object{blip_ending}.pkl')
        if (evaluate_from_scratch or not os.path.exists(path_full)
                or not os.path.exists(path_normalized)
                or not os.path.exists(path_per_object)):
            print("Eval results do not exist - creating them")
            df_results = pd.DataFrame(columns=['sentence', 'object_type', 'object_name', 'score', 'order', 'layer', 'index'])

            df_results_per_object = (
                pd.DataFrame(columns=['sentence', 'object_type', 'object_name', 'scores', 'order']))

            normalized_results_per_object = (
                pd.DataFrame(columns=['sentence', 'object_type', 'object_name', 'scores', 'order']))

            print("Starting to evaluate results")
            compositional_items_dataset = self.get_dataset()
            if compositional_items_dataset is None:
                if iterate_by == 'folder':
                    folder_name = params.folder_name
                    if not os.path.exists(folder_name):
                        raise Exception(f"Folder {folder_name} does not exist")
                    folder_names = os.listdir(folder_name)
                    folder_names = [folder for folder in folder_names if not folder.endswith('.txt') and not
                    folder.endswith('.pkl') and not folder.endswith('plots')]
                    # create a file and print into it all the folder names seperate by \n
                    with open(os.path.join(folder_name, 'folder_names.txt'), 'w') as f:
                        for folder in folder_names:
                            f.write(f'{folder}\n')
                    params.input_filename = os.path.join(folder_name, 'folder_names.txt')
                # TODO add a set_type and create a dataset according to it
                self.create_dataset(set_type=set_type, params=params)
                compositional_items_dataset = self.get_dataset()
                print("Number of compositional items: ", len(compositional_items_dataset))
            for compositional_item in tqdm.tqdm(compositional_items_dataset):
                curr_folder = os.path.join(self.main_folder_name, compositional_item.full_sentence)
                if not os.path.exists(curr_folder):
                    print("Cant find folder - try to add ' ' to the end of the folder name")
                    curr_folder = f'{curr_folder} '
                    if not os.path.exists(curr_folder):
                        print("Cant find folder - try to strip the folder name")
                        curr_folder = curr_folder.strip()
                        print("Folder does not exist (after strip): ", curr_folder)
                        continue
                else:
                    print("curr_folder: ", curr_folder)
                images_for_clip = []
                if args.model_key == 'sd1.4':
                    num_of_layers = 13
                elif args.model_key == 'sd2.1':
                    num_of_layers = 24
                else: # v1 - deep floyd
                    num_of_layers = 25
                # num_of_layers = 13 if args.model_key == 'sd1.4' else 25
                for layer in range(num_of_layers):
                    for index in range(self.img_num):
                        curr_img_path = os.path.join(curr_folder, args.model_key, 'encoder_full_direct',
                                                     'all_images', f'full_layer_{layer}_idx_{index}.png')
                        try:
                            img = Image.open(curr_img_path)
                        except:
                            print("failed to open image: ", curr_img_path)
                            try:
                                curr_img_path = os.path.join(curr_folder, args.model_key, 'encoder_full_direct',
                                                             'all_images', f'layer_{layer}_idx_{index}.png')
                                img = Image.open(curr_img_path)
                            except:
                                print("failed to open image: ", curr_img_path)
                                continue
                        images_for_clip.append(curr_img_path)
                sentences_dict = self.get_texts_for_clip_scores(compositional_item=compositional_item,
                                                                test_type=params.test_type)


                df_clip_scores = self.get_clip_scores(img_paths=images_for_clip, sentences_dict=sentences_dict,
                                                      softmax=False, is_batch=True,
                                                      full_sentence=compositional_item.full_sentence,
                                                      order=compositional_item.order, use_blip=params.use_blip)

                df_results = pd.concat([df_results, df_clip_scores], ignore_index=True)
                print("df_results shape: ", df_results.shape)
                # generate plot
                print("Generating plot")
                blip_flag = f'_blip_{args.blip_model_size}' if params.use_blip else ''
                print("Using blip: ", blip_flag)
                path = os.path.join(args.folder_name, f'plots{blip_flag}', 'per_sentence_plots',
                                    compositional_item.full_sentence, args.model_key)
                # check if path is too long
                if len(path) > 255:
                    print("Path is too long - shorten it")
                    path = os.path.join(args.folder_name, f'plots{blip_flag}', 'per_sentence_plots',
                                        f'{compositional_item.full_sentence[:100]}', args.model_key)

                preprocess_descriptions = self.create_plot(model_key= args.model_key, df_results=df_results, full_sentence=compositional_item.full_sentence,
                                                path=path, set_type=set_type, create_plot_per_sentence=create_plot_per_sentence)
                descriptions = preprocess_descriptions['descriptions']
                normalized_descriptions = preprocess_descriptions['normalized_descriptions']
                for description_key, description_value in descriptions.items():
                    print("Description key: ", description_key)
                    print("current shape:", df_results_per_object.shape)
                    df_results_per_object = pd.concat([df_results_per_object,
                                                        pd.DataFrame([[compositional_item.full_sentence,
                                                                       description_value['object_type'],
                                                                       description_value['object_name'],
                                                                       description_value['scores'],
                                                                       compositional_item.order]],
                                                                     columns=['sentence',
                                                                              'object_type',
                                                                              'object_name',
                                                                              'scores',
                                                                              'order'
                                                                              ])], ignore_index=True)
                for description_key, description_value in normalized_descriptions.items():
                    normalized_results_per_object = pd.concat([normalized_results_per_object,
                                                        pd.DataFrame([[compositional_item.full_sentence,
                                                                         description_value['object_type'],
                                                                            description_value['object_name'],
                                                                            description_value['scores'],
                                                                       compositional_item.order]],
                                                                     columns=['sentence', 'object_type', 'object_name',
                                                                              'scores', 'order'
                                                                              ])], ignore_index=True)
        else:
            print("Eval results already exist - loading them")
            df_results = pd.read_pickle(path_full)
            df_results_per_object = pd.read_pickle(path_per_object)
            normalized_results_per_object = pd.read_pickle(path_normalized)

        if len(df_results) == 0:
            print("No results were found")
        blip_flag = f'_blip_{args.blip_model_size}' if params.use_blip else ''
        path = os.path.join(params.folder_name, f'plots{blip_flag}', args.model_key)
        print("Creating aggregation plot")
        print("DF per object results shape:", df_results.shape)

        self.create_aggregation_plot(df_results=df_results_per_object, path=path, args=args, is_normalized=False)
        # self.create_aggregation_plot(df_results=df_results_per_object, path=path, is_normalized=True)
        # self.create_aggregation_plot(df_results=normalized_results_per_object, path=path, is_normalized=False)
        self.create_aggregation_plot(df_results=normalized_results_per_object, path=path, args=args, is_normalized=True)
        print("Done creating the df")
        return {
            'df_results': df_results,
            'df_results_per_object': df_results_per_object,
            'normalized_results_per_object': normalized_results_per_object
        }

    def calculate_openclip_score(self, prompts, img_path=None, img=None, softmax=True, is_list=False, is_blip=False):
        pil_images = []
        processed_images = []
        if is_list:
            if img is None:
                for im_path in img_path:
                    pil_images.append(Image.open(im_path))
                for i in range(len(pil_images)):
                    if not is_blip:
                        processed_images.append(self.preprocess(pil_images[i]).unsqueeze(0))
                    else:
                        processed_images.append(pil_images[i])

            else:
                processed_images = img
        else:
            if img is None:
                img = Image.open(img_path)
            if not is_blip:
                image = self.preprocess(img).unsqueeze(0)
            else:
                image = img
            # print(prompts)
        text = self.tokenizer(list(prompts))

        if is_blip:
            # create a tensor of the shape number of images on prompts
            res = torch.zeros((len(processed_images), len(prompts)))
            pre_prompt = "Does the following caption accurately describe the above image?"
            after_prompt = "Respond with yes or no"
            # pre_prompt = "Please respond with yes or no. Is the image showing the following? "
            # prompts = [pre_prompt + prompt for prompt in prompts]
            for img_idx, image in enumerate(processed_images):
                for prompt_idx, prompt in enumerate(prompts):
                    prompt = f'{pre_prompt} {prompt}. {after_prompt}'
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    inputs = self.blip_processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
                    self.blip_model = self.blip_model.to(device)
                    generated_ids = self.blip_model.generate(**inputs)
                    generated_text = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    # print("Q:", prompt,"\nA:", generated_text)
                    if generated_text.lower() == 'no':
                        score = 0
                    elif generated_text.lower() == 'somewhat':
                        score = 50
                    elif generated_text.lower() == 'yes':
                        score = 100
                    else:
                        print("prompt: ", prompt)
                        print("Error: generated_text is not yes or no")
                        print("generated_text: ", generated_text)
                        score = -100
                    res[img_idx, prompt_idx] = score

                    # plot the image with the prompt and score
                    blip_test_path = os.path.join(self.main_folder_name, 'blip_test')
                    if not os.path.exists(blip_test_path):
                        os.makedirs(blip_test_path)
                    plt.imshow(image)
                    plt.title(f'Prompt: {prompt}\nScore: {score}')
                    plt.savefig(os.path.join(blip_test_path,
                                                f'{prompt[:min(len(prompt), 10)]}_{prompt_idx}_{img_idx}.png'))
                    # print("Blip test - Saved plot in path: ", os.path.join(blip_test_path,
                    #                             f'{prompt[:min(len(prompt), 10)]}_{prompt_idx}_{img_idx}.png'))
                    plt.close()
                    # res[img_idx, :]
                    # = score



        else:
            images = torch.cat(processed_images, dim=0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                if softmax:
                    res = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                else:
                    res = image_features @ text_features.T
        return res


    def final_eval(self, df_results_full, folder_name, set_type):
        print("Starting final eval")
        # plot the clip scores graph for each sentence
        sentence_history = []
        for index, row in df_results_full.iterrows():
            # print(f"Index: {index}")
            sentence = row['sentence']
            # print("sentence: ", sentence)
            if sentence not in sentence_history:
                continue
            sentence_history.append(sentence)
            df_sentence = df_results_full[df_results_full['sentence'] == sentence]
            df_sentence = df_sentence.sort_values(by=['layer', 'index'])
            df_sentence = df_sentence.reset_index(drop=True)
            sns.set_theme()  # apply default seaborn theme
            sns.color_palette('colorblind')
            sns.lineplot(data=df_sentence, x='index', y='score', color='blue', marker='o')  # use seaborn lineplot
            plt.title(sentence)
            plt.xlabel('index')
            plt.ylabel('score')
            plt.savefig(os.path.join(folder_name, f'{sentence}.png'))
            plt.savefig(os.path.join(folder_name, f'{sentence}.pdf'), format='pdf', bbox_inches='tight')
            plt.close()
            print("Saved plot in path: ", os.path.join(folder_name, f'{sentence}.png'))
            # plt.clf()
            # plt.plot(df_sentence['score'])
            # plt.title(sentence)
            # plt.xlabel('index')
            # plt.ylabel('score')
            # plt.savefig(os.path.join(folder_name, f'{sentence}.png'))
            # plt.close()
            # print("Saved plot in path: ", os.path.join(folder_name, f'{sentence}.png'))


def flip_gender(sentence_list):
    flipped_sentences = []
    for sentence in sentence_list:
        if 'female' in sentence:
            sentence = sentence.replace('female', 'male')
        elif 'male' in sentence and 'female' not in sentence:
            sentence = sentence.replace('male', 'female')
        flipped_sentences.append(sentence)
    return flipped_sentences




def get_params(set_type, variation=None):
    if set_type == 'woman_wearing':
        return Box({
            'places': ["soccer stadium"], #  , "gym", "office"],
            'colors': ["green", "yellow"],
            'small_objects': ["glasses", "watch"], #, "hat"],
            'big_objects': ['suit', 'dress'],
            'orders': ['woman_place_big_small', 'place_woman_big_small',
                       'woman_place_small_big', 'small_first', 'big_first']
        })
    elif set_type == 'animal_acts':
        return Box({
            'animals': ["bear"], #, "mouse", "rabbit", "dog"],
            'objects': ["glasses", "shoes"],
            'acts': ["running", "sleeping"],
            'colors': ["blue", "white", "green"], #, "yellow"],
            'orders': ['animal_act_object', 'act_animal_object', 'object_animal_act', 'object_act_animal']
        })
    elif set_type == 'shapes':
        return Box({
            'shapes': ["circle", "square"], #, "triangle", "rectangle"],
            'colors': ["blue", "red", "yellow"],
            'surfaces': ["table", "floor"],
            'orders': ['shapes_surface', 'surface_shapes']
        })
    elif set_type == 'animal':
        if variation == 'uncommon':
            return Box({'animals':
                ['blobfish', 'narwhal', 'axolotl', 'okapi', 'fossa', 'numbat', 'gharial', 'quokka', 'quoll',
                 'pangolin', 'tapir', 'dugong', 'wombat', 'platypus', 'aardvark', 'aye-aye', 'binturong',
                 'capybara', 'coelacanth', 'dik-dik', 'echidna', 'frilled shark', 'gerenuk',
                 'hoatzin', 'jerboa', 'kiwi bird', 'lamprey', 'olm', 'saiga antelope',
                 'tarsier', 'umbrellabird', 'vaquita', 'wolffish', 'xenops', 'yak', 'quagga']
                })
        else:
            return Box({
                'animals': ["bear", "mouse", "rabbit", "dog", "cat", "elephant", "giraffe", "zebra",
                            "horse", "cow", "sheep", "pig", "chicken", "duck", "goose", "fish", "bird",
                            "turtle", "snake", "frog", "lizard", "crocodile", "alligator", "shark", "whale"]
            })

    elif set_type == 'animal_object':
        return Box({
            'animals': ["bear", "mouse", "rabbit", "dog", "cat", "elephant", "giraffe", "zebra",
                        "horse", "cow"],
            'objects': ["glasses", "shirt", "shoes", "hat", "scarf"],
            'orders': ['animal_object', 'object_animal']
        })
    elif set_type == 'object_size':
        print("Variation: ", variation)
        if variation is None or variation == 'None':
            print("Variation is None")
            big_objects = ['bear', 'elephant', 'giraffe', 'horse', 'cow']
            small_objects = ['mouse', 'rabbit', 'dog', 'cat', 'chicken', 'duck', 'goose', 'fish', 'bird', 'frog']
        elif variation == 'things':
            print("Variation is things")
            big_objects = ['moon', 'tree', 'house', 'bus', 'truck']
            small_objects = ['cube', 'key', 'ball', 'cup', 'box', 'ring', 'phone', 'book', 'pen', 'chair']
        else:
            raise NotImplementedError

        return Box({
            'big_objects': big_objects,
            'small_objects': small_objects,
            'orders': ['big_small', 'small_big']
        })
    elif set_type == 'animal_popularity':
        popular_animals = ['bear', 'mouse', 'rabbit', 'dog', 'cat'] #, 'elephant', 'giraffe', 'zebra', 'horse', 'cow',
                           # 'sheep', 'pig', 'chicken', 'duck', 'goose', 'fish', 'bird'
        unpopular_animals = ['blobfish', 'narwhal', 'axolotl', 'okapi' , 'fossa']
            #, 'numbat', 'gharial', 'quokka', 'quoll', 'tarsier']
                 # 'pangolin', 'tapir', 'dugong', 'wombat', 'platypus', 'aardvark', 'aye-aye', 'binturong',
                 # 'capybara', 'coelacanth', 'dik-dik', 'echidna', 'frilled shark', 'gerenuk',
                 # 'hoatzin', 'jerboa', 'kiwi bird', 'lamprey', 'olm', 'saiga antelope',
                 # 'tarsier',]
        return Box({
            'popular_animals': popular_animals,
            'unpopular_animals': unpopular_animals,
            'orders': ['popular_unpopular', 'unpopular_popular']
        })
    elif set_type == 'natural':
        return Box({
            'base_objects': ['banana', 'apple', 'orange', 'lemon', 'grape'],
            'natural_objects': ['pear', 'peach', 'pineapple', 'watermelon', 'coconut'],
            'unnatural_objects': ['chair', 'moon', 'elephant', 'rocket', 'truck'],
            'orders': ['base_natural', 'base_unnatural']
        })
    elif set_type == 'celebs':
        return Box({
            'celebs': ['angelina jolie', 'brad pitt', 'tom cruise', 'tom hanks', 'jennifer aniston',
            'julia roberts', 'leonardo dicaprio', 'matt damon', 'meryl streep', 'nicole kidman',
            'robert de niro', 'scarlett johansson', 'will smith', 'william shakespeare', 'jim carrey',
            'Michael Jackson', 'Elvis Presley', 'Marilyn Monroe', 'Albert Einstein', 'Abraham Lincoln',
            'John F. Kennedy', 'Martin Luther King', 'Nelson Mandela', 'Winston Churchill', 'Bill Gates',
            'Muhammad Ali', 'Mahatma Gandhi', 'Barack Obama', 'Steve Jobs', 'Stephen Hawking', 'Shaquille ONeal',
            'David Beckham', 'Michael Jordan', 'Tiger Woods', 'Roger Federer', 'Cristiano Ronaldo', 'Lionel Messi'
            ]
        })
    elif set_type == 'things':
        return Box({
            'things': ['banana', 'apple', 'orange', 'lemon', 'grape', 'pear', 'peach', 'pineapple', 'watermelon',
                       'coconut', 'chair', 'moon', 'elephant', 'rocket', 'truck', 'cube', 'key', 'ball', 'cup',
                       'box', 'ring', 'phone', 'book', 'pen', 'chair', 'table', 'floor', 'can', 'bottle', 'fork',
                        'spoon', 'knife', 'plate', 'bowl', 'cup', 'glass', 'chair', 'couch', 'computer', 'bed',
                        'lamp', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'car'
                          ]
        })
    elif set_type == 'thing_color':
        return Box({
            'things': ['chair', 'table', 'floor', 'can', 'bottle', 'fork',
                        'spoon', 'knife', 'plate', 'bowl', 'cup', 'glass', 'chair', 'couch', 'computer', 'bed',
                        'lamp', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'car'
                          ],
            'colors': ['red', 'blue', 'green', 'yellow']
        })
    elif set_type == 'two_things_color':
        return Box({
            'things': ['dog', 'rocket', 'robot', 'dragon', 'cat'],
            'colors': ['red', 'blue', 'green']
        })

    elif set_type == 'gender_bias':
        df_gender = pd.read_csv('inputs/TIMED_gender_test_set.csv')
        return Box({
            'sentences': list(df_gender['new']),
            'biased_sentences': flip_gender(list(df_gender['new'])),
            'unbiased_sentences': list(df_gender['old'])
        })
    elif set_type == 'general_bias':
        df_general = pd.read_csv('inputs/TIMED_test_set.csv')
        return Box({
            'sentences': list(df_general['new']),
            'unbiased_sentences': list(df_general['old'])
        })
    elif set_type == 'relations':
        return Box({
            'relations': ['on top of', 'underneath', 'on the left of', 'on the right of'],
            'objects': ['cat', 'strawberry', 'sushi', 'brain', 'beach hat', 'bike']
        })
    elif set_type == 'counting':
        return Box({
            'objects': ['cat', 'brain', 'bike', 'hat'],
            'number_of_objects': ['two'] # ['four']
        })
    else:
        print("set_type is unknown or None")
        return Box({})


if __name__ == '__main__':
    """
    Generation:
    --generate should be passed if you want to generate images
    --folder_name is the folder that the images would e saved into. In would be in generations folder
    --set_type should be passed only if you want to generate a specific set of prompts that are generated from a template, 
    otherwise it should be None (the default)
    --input_filename should be passed only if you want to generate a specific set of prompts that are generated from a file,
    otherwise it should be None (the default) The prompts should be seperated by \n (see ABC-6K.txt for example)
    --number_of_inputs should be passed only if you want to generate a specific number of images, 
    otherwise it should be -1 (the default) If a positive number is passed, only the first number_of_inputs images would be generated
    --model_key should be passed only if you want to generate images from a specific model,
    otherwise it should be 'v1' (the default) v1 is Deep Floyd, the other ones are stable diffusion and stable diffusion XL
    
    Evaluation:
    --evaluate should be passed if you want to evaluate images
    --folder_name is the folder that the images are saved to. 
    --set_type should be passed only if you want to evaluate a specific set of prompts that are generated from a template,
    otherwise it should be None (the default)
    
    """
    print("Running experiment")

    parser = argparse.ArgumentParser(description='Visualize hidden states of a T5 model')
    parser.add_argument('--folder_name', type=str, help='folder name',
                        default='compositional')
    parser.add_argument('--set_type', type=str, help='set type',
                        default=None,
                        choices=['woman_wearing', 'animal_acts', 'shapes', 'animal',
                                 'animal_object', 'object_size', 'natural', 'celebs', 'things',
                                 'gender_bias', 'general_bias', 'thing_color', 'two_things_color',
                                 'animal_popularity', 'relations', 'counting', 'None'])
    parser.add_argument('--generate', action='store_true',
                        help='generate images')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate images')
    # parser.add_argument('--make_graphs', action='store_false',
    #                     help='make graphs')
    parser.add_argument('--input_filename', type=str, help='input filename',
                        default=None)
    parser.add_argument('--number_of_inputs', type=int, help='number of inputs',
                        default=-1)
    parser.add_argument('--model_key', type=str, help='model key',
                        default='v1', choices=['v1', 'sd1.4', 'sd2.1', 'sdxl']), # , 'byt5'])
    parser.add_argument('--img_num', type=int, help='number of images per layer',
                        default=4)
    parser.add_argument('--test_type', type=str, help='test type',
                        default='None', choices=['distance_from_root', 'order', 'None'])
    parser.add_argument('--sentence_plot', action='store_true',
                        help='create plot per sentence')
    parser.add_argument('--variation', type=str, help='variation',
                        default=None, choices=['things', 'None', 'biased', 'uncommon'])
    parser.add_argument('--use_blip', action='store_true', help='use blip')
    parser.add_argument('--evaluate_from_scratch', action='store_true', help='evaluate from scratch')
    parser.add_argument('--blip_model_size', type=str, help='blip model size',
                        default='xl', choices=['xl', 'xxl'])
    parser.add_argument('--skip_all_layers', action='store_true', help='skip all layers')
    parser.add_argument('--start_layer', type=int, help='start layer',
                        default=0)
    parser.add_argument('--end_layer', type=int, help='end layer',
                        default=None)
    parser.add_argument('--step_layer', type=int, help='step layer',
                        default=1)
    parser.add_argument('--explain_other_model', action='store_true', help='explain other model')
    parser.add_argument('--clean_pads', action='store_true', default=False,
                        help='do change the pads to <pad> tokens from the same layer')
    parser.add_argument('--per_token', action='store_true', default=False,
                        help='do change the pads to <pad> tokens from the same layer')

    args = parser.parse_args()
    print(args)

    sns.color_palette('colorblind')
    # fix_names()
    # print("Fixed names")

    set_type = args.set_type if args.set_type != 'None' else None
    added_args = get_params(set_type, variation=args.variation)
    added_args['folder'] = args.folder_name
    for one_arg in added_args.keys():
        setattr(args, one_arg, added_args[one_arg])


    iterate_by = 'folder' if set_type is None else 'name'
    args.folder_name = os.path.join('generations', args.folder_name)
    if not os.path.exists(args.folder_name):
        try:
            os.makedirs(args.folder_name)
        except OSError as error:
            print("Error: ", error)

    compositional_experiment = CompositionalExperiment(main_folder_name=args.folder_name, img_num=args.img_num,
                                                       blip_model_size=args.blip_model_size)
    if args.generate:
        compositional_experiment.run_experiment(set_type=set_type, params=args)

