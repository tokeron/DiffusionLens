import torch
from diffusers import  DPMSolverMultistepScheduler # StableDiffusionPipeline
import os
from diffusers_local.src.diffusers import (StableDiffusionPipeline, StableDiffusionXLPipeline,
                                           IFPipeline, IFSuperResolutionPipeline) # StableDiffusionPipeline
import matplotlib.pyplot as plt
import argparse


def make_grid(images, nrow=2):
    plt.figure(figsize=(20, 20))
    if len(images) > 4:
        print(f"len images: {len(images)}")
        nrow = (len(images) // 2 ) + 1
    for index, image in enumerate(images):
        plt.subplot(nrow, nrow, index+1)
        plt.imshow(image)
        plt.axis('off')
    # Adjust the spacing between the subplots
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)
    return plt


def stable_glass_sd(args_list, multiple_args=True):
    if multiple_args:
        args = args_list[0]
    else:
        args = args_list
        args_list = [args]
    if 'seed' not in args:
        args.seed = 42
    seed = args.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.manual_seed(seed)


    models_dict = {
        'sd2.1': 'stabilityai/stable-diffusion-2-1',
        'sdxl': 'stabilityai/stable-diffusion-xl-base-1.0',
        'sd1.4': 'CompVis/stable-diffusion-v1-4',
        'v1': 'DeepFloyd/IF-I-XL-v1.0'
    }

    if 'model_key' not in args:
        print("model key is default: sd2.1")
        args.model_key = 'sd2.1'
    else:
        print(f"model key is {args.model_key}")
    model_name = models_dict[args.model_key]


    if args.model_key == 'sdxl':
        print("Using SDXL model")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            generator=generator,
        )
    elif args.model_key == 'v1':
        print("Using IF model")
        pipe = IFPipeline.from_pretrained(model_name,
                                          variant="fp16",
                                          torch_dtype=torch.float16,
                                          generator=generator,
                                          start_layer=args.start_layer,
                                          end_layer=args.end_layer,
                                          step_layer=args.step_layer,
                                          )
        pipe = pipe.to("cuda")
        super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16,
            generator=generator,
        )
        super_res_1_pipe = super_res_1_pipe.to("cuda")

    elif args.model_key == 'sd1.4' or args.model_key == 'sd2.1':
        print("Using SD 2.1 or 1.4 model")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            # variant="fp16",
            # repo_type="huggingface",
            torch_dtype=torch.float16,
            # use_safetensors=True,
            generator=generator,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            step_layer=args.step_layer,
        )
    else:
        raise ValueError("Model key is not supported")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    for args in args_list:
        if args.model_key == 'sd2.1':
            last_layer = 23
        elif args.model_key == 'sd1.4':
            last_layer = 12
        elif args.model_key == 'sdxl':
            last_layer = 25
        elif args.model_key == 'v1':
            last_layer = 24

        path_to_last_image = os.path.join(args.main_folder_name, args.prompt, args.model_key,
                                           'encoder_full_direct', f'layer_{last_layer}.png')
        if os.path.exists(path_to_last_image):
            print(f"{path_to_last_image} exist :)")
            continue
        else:
            print(f"{path_to_last_image} does not exist")

        if 'output_folder' not in args and 'main_folder_name' in args:
            args.output_folder = args.main_folder_name

        if 'output_folder' not in args or not args.output_folder:
            print('output folder is default: outputs')
            args.output_folder = 'outputs'
        # folder_name = args.output_folder
        if 'img_num' not in args:
            args.img_num = 4
        img_num = args.img_num
        if 'input_filename' not in args or not args.input_filename:
            if not args.prompt:
                raise ValueError('Prompt must be provided if input folder is not provided')
            else:
                prompts = [args.prompt]
        else:
            with open(f"inputs/{args.input_filename}", 'r') as f:
                prompts = f.readlines()

        skip_layer_list = [None]
        if args.skip_all_layers:
            skip_layer_list += [[x] for x in range(2, last_layer - 1)]

        for prompt in prompts:
            print(f"prompt: {prompt}")
            if '\n' in prompt:
                prompt = prompt.replace('\n', '')
            output_folder = os.path.join(args.output_folder, prompt)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            if not model_name == 'sdxl':
                print("img_num", img_num)
                device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
                generator = torch.Generator(device.type).manual_seed(seed)
                for skip_layers in skip_layer_list:
                    if skip_layers is not None:
                        skip_layer_ending = '_skip_layers_' + str(skip_layers)
                        start_from = skip_layers[0] #  - 1
                        if start_from is None:
                            start_from = 0
                    else:
                        skip_layer_ending = ''
                        start_from = 0
                    if args.model_key == 'v1':
                        images = []
                        low_res_images, embeds_per_layer = pipe(prompt,
                                                                num_images_per_prompt=img_num,
                                                                generator=generator,
                                                                start_layer=args.start_layer,
                                                                end_layer=args.end_layer,
                                                                step_layer=args.step_layer,
                                                                explain_other_model=args.explain_other_model,
                                                                )

                        print(f"len embeds_per_layer: {len(embeds_per_layer)}")
                        for index, (low_res_images, (embeddings, negative_embeddings)) in (
                                enumerate(zip(low_res_images, embeds_per_layer))):
                            # print("+" * 100)
                            # print(f"images len: {len(low_res_images.images)}")
                            # print(f"(item in embeds_per_layer[0]) embeddings shape: {embeddings.shape}")
                            # print("+" * 100)
                            # embeddings, negative_embeddings = embeddings[:4, :, :], embeddings[4:, :, :]
                            print("+" * 100)
                            print(f"embeddings shape: {embeddings.shape}")
                            print(f"negative_embeddings shape: {negative_embeddings.shape}")
                            print("+" * 100)
                            curr_image = super_res_1_pipe(
                                image=low_res_images.images,
                                prompt_embeds=embeddings,
                                negative_prompt_embeds=negative_embeddings,
                                num_images_per_prompt=1,
                            )
                            images.append(curr_image)
                    else:
                        images = pipe(prompt,
                                      num_images_per_prompt=img_num,
                                      generator=generator,
                                      skip_layers=skip_layers,
                                      start_layer=args.start_layer,
                                      end_layer=args.end_layer,
                                      step_layer=args.step_layer,
                                      explain_other_model=args.explain_other_model,
                                      per_token=args.per_token,
                        )


                    print(f"The number of images is {len(images)}")

                    text_input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

                    token_index_counter = 0
                    for token_index in range(len(text_input_ids[0])):
                        if args.per_token:
                            per_token_ending = f'_token_{token_index}'
                        if token_index_counter > 0 and not args.per_token:
                            break
                        for index, image in enumerate(images):
                            output_folder_per_image = os.path.join(output_folder, args.model_key,
                                                                   f'encoder_full_direct{skip_layer_ending}', 'all_images')
                            if not os.path.exists(output_folder_per_image):
                                os.makedirs(output_folder_per_image)
                            output_folder_for_grid = os.path.join(output_folder, args.model_key,
                                                                    f'encoder_full_direct{skip_layer_ending}')
                            if not os.path.exists(output_folder_for_grid):
                                os.makedirs(output_folder_for_grid)
                            for num in range(img_num):
                                image.images[num].save(f'{output_folder_per_image}/'
                                                       f'full_layer_'
                                                       f'{args.start_layer + start_from + index * args.step_layer}'
                                                       f'_idx_{num}{per_token_ending}.png')
                            nrow = 1 if img_num == 1 else 2
                            plot = (make_grid(image.images, nrow=nrow))
                            plot.savefig(f'{output_folder_for_grid}/'
                                         f'layer_{args.start_layer + start_from + index * args.step_layer}{per_token_ending}.png')
                            plot.close()
                        token_index_counter += 1
            else:
                raise ValueError("SDXl diffusion lens is not implemented")



if __name__ == '__main__':
    print("Running experiment")
    parser = argparse.ArgumentParser(description='Visualize hidden states of stable diffusion')

    parser.add_argument('--prompt', type=str,
                        default='Bill Gates',
                        help='prompt to visualize')
    parser.add_argument('--input_filename', type=str,
                        help='input filename for texts - if not provided: prompts must be provided')
    parser.add_argument('--output_folder',
                        type=str,
                        help='output folder name')
    parser.add_argument('--model_key',
                        type=str,
                        help='model key in the models dictionary',
                        default='sd2.1',
                        choices=['sd2.1', 'sdxl', 'sd1.4', 'v1'])
    parser.add_argument('--img_num',
                        type=int,
                        default=4,
                        help='Number of images to generate for each layer')
    parser.add_argument('--set_type',
                        type=str,
                        help='set type',
                        default='man',
                        choices=['man', 'animal', 'shapes', 'None'])
    parser.add_argument('--generate',
                        action='store_true',
                        help='generate images')
    parser.add_argument('--evaluate',
                        action='store_true',
                        help='evaluate images')
    parser.add_argument('--make_graphs',
                        action='store_false',
                        help='make graphs')
    parser.add_argument('--main_folder_name',
                        type=str,
                        help='main folder name',
                        default='outputs')
    parser.add_argument('--per_token',
                        action='store_true',
                        help='per token')


    args = parser.parse_args()
    stable_glass_sd(args, multiple_args=False)
