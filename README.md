
# Chinese InstructBLIP

+ To test and enable Chinese interaction capability for [InstructBLIP](https://arxiv.org/abs/2305.06500), we have added the [Randeng](https://huggingface.co/IDEA-CCNL/Randeng-Deltalm-362M-En-Zh) translation model before its input and after its output. Example code on Colab:<a href="https://colab.research.google.com/drive/1s-yy6POjNiQ6qzv4t8Q7Vnk22uUrVA7G?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


+ We have also tested the Chinese interaction capability of [VisualGLM](https://github.com/THUDM/VisualGLM-6B) ([ChatGLM-6b](https://github.com/THUDM/VisualGLM-6B)) and InstructBLIP ([Vicuna-7b](https://lmsys.org/blog/2023-03-30-vicuna/)). Partial test results can be found in [questions.md](questions.md) file.

#### online demo
<div align="center">
  <img src="./img/run_demo.gif" alt="Image"/>
</div>

+ We have also replaced [run_demo.py](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/run_demo.py) of the InstructBLIP project with our [demo_InstructBLIP_WithTranslate.py](./demo_InstructBLIP_WithTranslate.py) to enable online Chinese inquiry functionality.
