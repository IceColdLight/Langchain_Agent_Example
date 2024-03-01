# Session 1 - Documentation

Here is the document I created for you: https://docs.google.com/document/d/1z9T58MN5ik2ZFx5WPlrYNHYTu00LqYmrNmfNvo4m6Gg/edit?usp=sharing

# Installation

## Python
Install python 3.10

## Install CONDA

This will make managing dependencies with python a lot easier. Once you have it installed we will load an environment (packages, project name, etc.) from file:
```
conda env create -f environment.yml
conda activate langchain_test
```

## API or GGUF oR GPTQ?

This project uses the Mistral API, but I've written you a small guide that shows you how to use LLMs locally based on GGUF (even with GPU) and I've left some code in the files as reference. As you have a 4090 you might even be able to use GPTQ formats. However, be prepared for lots of headaches, as it's quite tricky getting this setup. Good luck!

## GGUF Guide (CPU with GPU offloading)

1. Install the relevant CUDA Toolkit version for your GPU from NVIDIA (https://en.wikipedia.org/wiki/CUDA#GPUs_supported). I use 11.4 for now.
2. Install cmake and Visual Studio with C++ Build Tools
3. Install pytorch from their website (has to be 11.8 because of pytorch, but works as its the same major version as 11.4): conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
4. pip3 install chardet (might be missing)
5. Test if CUDA works: print(torch.cuda.is_available())

## LLM Setup
We use Mixtral-8x7B-instruct-v.01 as it's the most performant small model. So in order to run it we need a quantised model. We use this 4 BITQ model: https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF (https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/blob/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf) which is a good trade-off of size and quality. 5 BIT works too, it's just very slow at inference for my system.

1. Download the model file
2. Since we use a quantisised GGUF file and Mistral is a finetuned LLAMA model, we can (thank god) use LLAMA-CPP to load the model which is supported by Langchain. In order to have GPU support here as well, we need to build it manually. See here on how to do that:

    i. https://python.langchain.com/docs/integrations/llms/llamacpp#installation-with-windows
    ii. If the build fails here are some solutions:
        - Set these 3 env variables like this or GPU use will NOT work:
            $env:CUDACXX="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe"
            $env:FORCE_CMAKE=1
            $env:CMAKE_ARGS="-DLLAMA_CUBLAS=ON"
        - pip install fsspec (should it be missing)
        - Sometimes VisualStudio's buildtools fail to connect CUDA properly: https://stackoverflow.com/a/56665992
        - This guide might also help: https://medium.com/@ryan.stewart113/a-simple-guide-to-enabling-cuda-gpu-support-for-llama-cpp-python-on-your-os-or-in-containers-8b5ec1f912a4

When running the code, you should see BLAS = 1 in the terminal. Offload as many layers to the GPU as possible. If BLAS = 0 then the build of llamacpp did not work properly.

## Making it faster (GGUF)
1. Startup - It's normal for the startup of the model to take a while. The whole model is loaded from disk into RAM or VRAM + RAM.
i. You can speed it up by loading the model from a SSD instead of a HDD
2. Inference - You have the choice between CPU + GPU and CPU only. As it turns out (see above), CPU only is faster for inference in our case.
i. You can improve speed by RAM clockspeed (how fast it can pass data to the CPU). 3600 is the sweetspot for my CPU. Unfortunately I only have 3200 Hrz, but I don't know how much of a difference that really makes.

Sidenotes:
1. The model will either fit into your RAM or it won't. There is no offloading to disk happening.
2. The higher the quantization, the A. more space will the model take up in RAM and B. more processing power will the model take. To find the sweetspot simply experiment (usually it has been found to be 4 bit tough for most models).

## Langchain
Now we have everything setup to start. Install langchain:

```
conda install langchain -c conda-forge
```

## FAISS Setup
If you want to experiment with RAG: https://python.langchain.com/docs/integrations/vectorstores/faiss

Install with pip, as conda doesn't have it:
```
pip install faiss-cpu
pip install tiktoken
```

## API Token

Finally, you need an API token for an API. I use Mistral, but you can also use OpenAI. Here is the link for Mistral: https://console.mistral.ai/

Install with pip, as conda doesn't have it:
```
pip install -qU langchain-core langchain-mistralai
```

# Run

```
conda activate langchain_test
python ./main.py
```

Should there be any other errors then there might be some modules missing that you need to install, but this should be pretty easy to do.

For questions, ask me anytime: sikluibenschaedl123@gmail.com