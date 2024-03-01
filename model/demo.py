import os
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from utils import load_json, load_yaml, merge_config, init_logger, get_rank
from demo import CustomTheme, ConversationalAgent


MODEL_CONFIG = "pegs/configs/common/pegs.yaml"
STICKER_EXAMPLES = "demo/stickers_for_demo.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="demo/config/demo.yaml")
    args = parser.parse_args()
    
    return args


def setup_seeds(config):
    seed = config.run.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main(args):
    # config
    demo_config = load_yaml(args.config)
    model_config = load_yaml(MODEL_CONFIG)
    config = merge_config([demo_config, model_config])
    
    # seed
    setup_seeds(config)
    # logging
    init_logger(config)
    # sticker examples
    sticker_examples = load_json(STICKER_EXAMPLES)
    
    agent = ConversationalAgent(config)
    
    theme = CustomTheme()
    title = """
        <font face="Courier" size=5>
            <center><B>StickerConv: Generating Multimodal Empathetic Responses from Scratch</B></center>
        </font>
    """
    language = """Language: English"""
    with gr.Blocks(theme) as demo_chatbot:
        gr.Markdown(title)
        # gr.HTML(article)
        gr.Markdown(language)
        
        with gr.Row():
            with gr.Column(scale=3):
                start_btn = gr.Button("Start Chat", variant="primary", interactive=True)
                clear_btn = gr.Button("Clear Context", interactive=False)
                undo_btn = gr.Button("undo", interactive=False)
                image = gr.Image(type="pil", interactive=False)
                
                with gr.Accordion("Generation Settings", open=False):
                    do_sample = gr.Radio([True, False],
                                        value=True,
                                        interactive=True,
                                        label="whether to do sample")
                    
                    top_p = gr.Slider(minimum=0, maximum=1, step=0.1,
                                      value=0.7,
                                      interactive=True,
                                      label='top-p value',
                                      visible=True)
                    
                    temperature = gr.Slider(minimum=0, maximum=1, step=0.1,
                                            value=0.7,
                                            interactive=True,
                                            label='temperature',
                                            visible=True)
                    
                    num_inference_steps = gr.Slider(minimum=0, maximum=100, step=1,
                                            value=50,
                                            interactive=True,
                                            label='num inference steps',
                                            visible=True)
                    
                    guidance_scale = gr.Slider(minimum=0, maximum=15, step=0.1,
                                            value=7.5,
                                            interactive=True,
                                            label='guidance scale',
                                            visible=True)
                    negativate_prompt = gr.Textbox(label='Negative prompt',
                                                   value='comic',
                                                   interactive=True)
                    
            with gr.Column(scale=7):
                chat_state = gr.State([])  # message, context, pixel_values_list
                chatbot = gr.Chatbot(label='PEGS', height=800, avatar_images=((os.path.join(os.path.dirname(__file__), 'demo/user.png')), (os.path.join(os.path.dirname(__file__), "demo/bot.png"))))
                text_input = gr.Textbox(label='User', placeholder="Please click the <Start Chat> button to start chat!", interactive=False)
               
        with gr.Row():
            example_happiness_sticker = gr.Examples(examples=sticker_examples["Happiness"], inputs=image, label="happiness")
            example_neutral_sticker = gr.Examples(examples=sticker_examples["Neutral"], inputs=image, label="neutral")
            example_sadness_sticker = gr.Examples(examples=sticker_examples["Sadness"], inputs=image, label="sadness")
        with gr.Row():
            example_surprise_sticker = gr.Examples(examples=sticker_examples["Surprise"], inputs=image, label="surprise")
            example_anger_sticker = gr.Examples(examples=sticker_examples["Anger"], inputs=image, label="anger")
            example_fear_sticker = gr.Examples(examples=sticker_examples["Fear"], inputs=image, label="fear")
        with gr.Row():
            example_disgust_sticker = gr.Examples(examples=sticker_examples["Disgust"], inputs=image, label="disgust")
                
        start_btn.click(agent.start_chat, [chat_state], [text_input, start_btn, clear_btn, image, chat_state])
        clear_btn.click(agent.restart_chat, [chat_state], [chatbot, text_input, start_btn, clear_btn, image, chat_state], queue=False)
        undo_btn.click(agent.undo, [chatbot, chat_state], [text_input, chatbot, chat_state])
        text_input.submit(
            agent.respond,
            inputs=[text_input, image, chatbot, do_sample, top_p, temperature, num_inference_steps, guidance_scale, negativate_prompt, chat_state], 
            outputs=[text_input, image, chatbot, chat_state, undo_btn]
        )
        
    demo_chatbot.launch(share=True, server_name="127.0.0.1", server_port=1081)
    demo_chatbot.queue()
    

if __name__ == "__main__":
    args = parse_args()

    main(args)
