import gradio as gr
from lavis.models import load_model_and_preprocess
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-name", default="blip2_vicuna_instruct")
    parser.add_argument("--model-type", default="vicuna7b")
    args = parser.parse_args()

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=3):
                image_input = gr.Image(type="pil")

                prompt_textbox = gr.Textbox(label="Prompt:", placeholder="prompt", lines=2)

                translate_zh2en_button = gr.Button("Chinese to English")

                submit_button = gr.Button("RUN")

                out_textbox = gr.Textbox(label="Out:", placeholder="answer", lines=2)
                translate_en2zh_button = gr.Button("English to Chinese")
                # translate_zh2en_button2 = gr.Button("Chinese to English")
            with gr.Column(scale=1):
                min_len = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=1,
                    step=1,
                    interactive=True,
                    label="Min Length",
                )

                max_len = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=250,
                    step=5,
                    interactive=True,
                    label="Max Length",
                )

                sampling = gr.Radio(
                    choices=["Beam search", "Nucleus sampling"],
                    value="Beam search",
                    label="Text Decoding Method",
                    interactive=True,
                )

                top_p = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    interactive=True,
                    label="Top p",
                )

                beam_size = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    interactive=True,
                    label="Beam Size",
                )

                len_penalty = gr.Slider(
                    minimum=-1,
                    maximum=2,
                    value=1,
                    step=0.2,
                    interactive=True,
                    label="Length Penalty",
                )

                repetition_penalty = gr.Slider(
                    minimum=-1,
                    maximum=3,
                    value=1,
                    step=0.2,
                    interactive=True,
                    label="Repetition Penalty",
                )

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        print('Loading model...')

        model, vis_processors, _ = load_model_and_preprocess(
            name=args.model_name,
            model_type=args.model_type,
            is_eval=True,
            device=device,
        )
        # model, vis_processors, _ = None,None,None
        print('Loading InstructBLIP model done!')
        print('Loading model...')
        # load Randeng translate model
        from fengshen.models.deltalm.modeling_deltalm import DeltalmForConditionalGeneration
        from transformers import AutoTokenizer

        model_translate = DeltalmForConditionalGeneration.from_pretrained("IDEA-CCNL/Randeng-Deltalm-362M-Zh-En").to(
            device)
        model_translate_en2zh = DeltalmForConditionalGeneration.from_pretrained(
            "IDEA-CCNL/Randeng-Deltalm-362M-En-Zh").to(
            device)
        tokenizer_translate = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
        # model_translate = None
        # tokenizer_translate = None
        print('Loading IDEA-CCNL/Randeng-Deltalm-362M model done!')


        def translate_fn(prompt_textbox):
            inputs = tokenizer_translate(prompt_textbox, max_length=512, return_tensors="pt").to(device)
            generate_ids = model_translate.generate(inputs["input_ids"], max_length=512)
            en_prompt_textbox = tokenizer_translate.batch_decode(generate_ids, skip_special_tokens=True,
                                                                 clean_up_tokenization_spaces=False)[0]

            return en_prompt_textbox


        def translate_fn_En2Zh(prompt_textbox):
            inputs = tokenizer_translate(prompt_textbox, max_length=512, return_tensors="pt").to(device)
            generate_ids = model_translate_en2zh.generate(inputs["input_ids"], max_length=512)
            en_prompt_textbox = tokenizer_translate.batch_decode(generate_ids, skip_special_tokens=True,
                                                                 clean_up_tokenization_spaces=False)[0]

            return en_prompt_textbox


        translate_zh2en_button.click(
            translate_fn,
            show_progress=True,
            inputs=[prompt_textbox],
            outputs=prompt_textbox,
        )

        translate_en2zh_button.click(
            translate_fn_En2Zh,
            show_progress=True,
            inputs=[out_textbox],
            outputs=out_textbox,
        )


        def inference(image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p,
                      decoding_method):
            use_nucleus_sampling = decoding_method == "Nucleus sampling"
            print(image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p,
                  use_nucleus_sampling)
            image = vis_processors["eval"](image).unsqueeze(0).to(device)

            samples = {
                "image": image,
                "prompt": prompt,
            }

            output = model.generate(
                samples,
                length_penalty=float(len_penalty),
                repetition_penalty=float(repetition_penalty),
                num_beams=beam_size,
                max_length=max_len,
                min_length=min_len,
                top_p=top_p,
                use_nucleus_sampling=use_nucleus_sampling,
            )

            return output[0]


        submit_button.click(
            inference,
            show_progress=True,
            inputs=[image_input, prompt_textbox, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p,
                    sampling],
            outputs=out_textbox
        )

    demo.launch(server_name='0.0.0.0',
                # ip for listening, 0.0.0.0 for every inbound traffic, 127.0.0.1 for local inbound
                server_port=8888,  # the port for listening
                show_api=False,  # if display the api document
                share=False,  # if register a public url
                inbrowser=False)