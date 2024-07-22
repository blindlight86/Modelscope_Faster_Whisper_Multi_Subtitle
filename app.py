import argparse

import gradio as gr

from utils import movie2audio,make_srt,merge_sub,make_tran_qwen2


initial_md = """

"""


def do_speech(video):

    return movie2audio(video)


def do_trans_video(model_type,video_path):

    return make_srt(video_path,model_type)

def do_trans_audio(model_type):

    return make_srt('./audio.wav',model_type)

def do_trans_qwen2(ollama_host, ollama_module,srt_path, lang):

    return make_tran_qwen2(ollama_host, ollama_module, srt_path, lang)

def do_srt_sin(video_path):

    return merge_sub(video_path,"./video.srt")

def do_srt_two(video_path):

    return merge_sub(video_path,"./two.srt")



with gr.Blocks() as app:
    gr.Markdown(initial_md)

    with gr.Accordion("视频处理(Video)"):
        with gr.Row():

            ori_video = gr.Video(label="请上传视频(Upload Video)")
        
            speech_button = gr.Button("提取人声(如果视频没有背景音也可以不做)Extract human voice (you don't have to do it if the video has no background sound)")

            speech_audio = gr.Audio(label="提取的人声(Extract voice)")

    
    speech_button.click(do_speech,inputs=[ori_video],outputs=[speech_audio])
    
    with gr.Accordion("转写字幕"):

        with gr.Row():
            with gr.Column():
                
                # model_type = gr.Dropdown(choices=["small","medium","large-v3","large-v2"], value="small", label="选择faster_Whisper模型/Select faster_Whisper model",interactive=True)

                model_type = gr.Textbox(label="填写faster_Whisper模型/Fill in the faster_Whisper model,也可以填写small,medium,large,large-v2,large-v3,模型越大，速度越慢，但字幕的准确度越高，酌情填写，用文本框是因为你可以填写其他huggingface上的开源模型地址",value="medium")


        with gr.Row():
            
            transcribe_button_whisper = gr.Button("Whisper视频直接转写字幕(Video direct rewriting subtitles)")

            transcribe_button_audio = gr.Button("Whisper提取人声转写字幕(Extract voice transliteration subtitles)")

            result1 = gr.Textbox(label="字幕結果(会在项目目录生成video.srt/video.srt is generated in the current directory)")

        transcribe_button_whisper.click(do_trans_video,inputs=[model_type,ori_video],outputs=[result1])


        transcribe_button_audio.click(do_trans_audio,inputs=[model_type],outputs=[result1])

    with gr.Accordion("Qwen2大模型字幕翻译"):
        with gr.Row():

            with gr.Column():
                ollama_host = gr.Textbox(label="Ollama大模型地址", value="http://127.0.0.1:11434")
                ollama_module = gr.Textbox(label="Qwen2大模型名称", value="qwen2:7b")
                srt_path_qwen2 = gr.Textbox(label="原始字幕地址，默认为项目目录中的video.srt,也可以输入其他路径",value="./video.srt")
                
            with gr.Column():
                trans_target_lang = gr.Dropdown(choices=["zh","en","ja","ko"], value="zh", label="选择目标语言/Select target language")
                trans_button_qwen2 = gr.Button("翻译字幕为目标语言/Translate subtitles into Target Language")

            result2 = gr.Textbox(label="翻译结果(会在项目目录生成two.srt/two.srt is generated in the current directory)")

        trans_button_qwen2.click(do_trans_qwen2,[ollama_host, ollama_module, srt_path_qwen2, trans_target_lang],outputs=[result2])

    with gr.Accordion("字幕合并"):
        with gr.Row():


            srt_button_sin = gr.Button("将单语字幕合并到视频/Merge monolingual subtitles into video")

            srt_button_two = gr.Button("将双语字幕合并到视频/Merge bilingual subtitles into video")

            result3 = gr.Video(label="带字幕视频")

    srt_button_sin.click(do_srt_sin,inputs=[ori_video],outputs=[result3])
    srt_button_two.click(do_srt_two,inputs=[ori_video],outputs=[result3])

parser = argparse.ArgumentParser()
parser.add_argument(
    "--server-name",
    type=str,
    default=None,
    help="Server name for Gradio app",
)
parser.add_argument(
    "--no-autolaunch",
    action="store_true",
    default=False,
    help="Do not launch app automatically",
)
args = parser.parse_args()

app.queue()
app.launch(inbrowser=True, server_name=args.server_name)
