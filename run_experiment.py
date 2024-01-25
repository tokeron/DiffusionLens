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
# import seaborn as sns

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
                    # pos = self.get_pos(params.prompt)
                    # current_item.set_pos(pos)
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
                    # pos = self.get_pos(prompt)
                    # current_item.set_pos(pos)
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



    def get_clip_scores(self, img_paths, sentences_dict, softmax, is_batch=False, full_sentence=None, order=None, use_blip=False):
        df_clip_scores = pd.DataFrame(columns=['sentence', 'object_type', 'object_name', 'score', 'order', 'layer', 'index'])
        sentences = list(sentences_dict.values())
        # scores = self.calculate_clip_score(img, sentences)

        scores = self.calculate_openclip_score(prompts=sentences, img_path=img_paths, softmax=softmax, is_list=is_batch,
                                                   is_blip=use_blip)

        print("Scores: ", scores)

        if not is_batch:
            img_paths = [img_paths]
        for image_idx, image_path in enumerate(img_paths):
            for sentence_key, sentence_val, score in (
                    zip(sentences_dict.keys(), sentences_dict.values(), scores[image_idx])):
                idx = image_idx % self.img_num
                layer = image_idx // self.img_num
                df_clip_scores = pd.concat([df_clip_scores,
                                            pd.DataFrame([[full_sentence, sentence_key, sentence_val,
                                                           score, order, layer, idx]],
                                                         columns=['sentence', 'object_type', 'object_name',
                                                                  'score', 'order', 'layer', 'index'])])


        return df_clip_scores

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



        if not os.path.exists(path):
            os.makedirs(path)


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


def get_params(set_type, variation=None):
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

    # sns.color_palette('colorblind')
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

