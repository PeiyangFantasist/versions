# 优化了UI设置，但是没有实现：1.保存密码 2.修复文件处理bug
import streamlit as st
import os
import requests
import easyocr
from PIL import Image
import io
import fitz  # PyMuPDF
import json
from groq import Groq
import openai
from openai import OpenAI
import time
import streamlit.components.v1 as components
import numpy as np

deepseek_url = "https://api.deepseek.com"
silicon_url = "https://api.siliconflow.cn/v1/"
groq_api_key = ""
# 初始化EasyOCR阅读器
reader = easyocr.Reader(['ch_sim', 'en'])

# 页面配置
st.set_page_config(page_title="ASTERISKY AI", layout="wide")

# 初始化session状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "context" not in st.session_state:
    st.session_state.context = None

# 保存


# 文件上传区域
uploaded_file = None

# 创建一个容器，用于放置标题和文件上传按钮
with st.container():
        st.markdown(
            """
            <div style="position: sticky; top: 0; background-color: white; z-index: 999;">
                <h1 style="margin: 0;">✳️ ASTERISKY STUDIO</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

# 侧边栏设置
with st.sidebar:
    st.header("API设置")
    selected_model = st.selectbox("选择模型", ["Groq", "Ollama", "DeepSeek", "SiliconCloud"])
    # 添加Ollama模型名称输入
    
    if selected_model == "Ollama":
        ollama_api_url = st.text_input("Ollama API地址", "http://localhost:11434/api/generate")
        ollama_model_name = st.text_input("Ollama模型名称", "deepseek-r1:8b")
        ollama_model_name = "deepseek-r1:8b"  # 默认值
    
    if selected_model == "SiliconCloud":
        silicon_model_name = st.text_input("SiliconCloud模型名称", "deepseek-ai/DeepSeek-R1")
        silicon_api_key = st.text_input("SiliconCloud API密钥", type="password")
        silicon_model_name = "deepseek-ai/DeepSeek-R1"  # 默认值

    if selected_model == "DeepSeek":
        deepseek_api_key = st.text_input("DeepSeek API密钥", type="password")
    if selected_model == "Groq":
        groq_api_key = st.text_input("Groq API密钥", type="password")
    
    # 清除上下文按钮
    st.header("功能")
    if st.button("🧹清除上下文"):
        st.session_state.context = None
        st.session_state.messages = []
        st.session_state.ocr_text = ""
        st.success("上下文已清除")
    
    # 文件上传区域
    uploaded_file = st.file_uploader("上传图片或PDF", type=["png", "jpg", "jpeg", "pdf"])
    st.header("参数设置")
    temperature = st.slider("温度", 0.0, 1.0, 0.7)
    system = st.text_area("系统提示", "")

# 把API密钥设置到环境变量中
os.environ["GROQ_API_KEY"] = groq_api_key

# 文件处理函数
def process_file(uploaded_file):
    if uploaded_file is None:
        return ""
    
    # 处理PDF
    if uploaded_file.type == "application/pdf":
        doc = fitz.open(stream=uploaded_file.read())
        text = ""
        for page in doc:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            result = reader.readtext(img)
            text += "\n".join([res[1] for res in result]) + "\n"
        return text

    # 处理图片
    else:
        img = Image.open(uploaded_file)
        result = reader.readtext(img)
        return "\n".join([res[1] for res in result])

# 流式API调用函数
def stream_ollama_response(prompt, model_name):
    data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "stream": True,
    }
    if st.session_state.context is not None:
        data["context"] = st.session_state.context

    try:
        with requests.post(
            ollama_api_url,
            json=data,
            stream=True,
            timeout=60
        ) as response:
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        yield chunk['response']
                    if chunk.get('done'):
                        st.session_state.context = chunk.get('context')

    except requests.exceptions.RequestException as e:
        yield f"请求发生错误：{str(e)}"
    except json.JSONDecodeError as e:
        yield f"解析响应数据时发生错误：{str(e)}"
    except Exception as e:
        yield f"发生未知错误：{str(e)}"

# Groq API调用函数（保持不变）
def call_groq_api(prompt, client):
    try:
        # 获取用户输入
        user_input = prompt

        # 调用 API 获取回复
        chat_completion = client.chat.completions.create(
            messages=st.session_state.messages,
            model="deepseek-r1-distill-llama-70b"
        )

        # 提取助手回复内容
        assistant_reply = chat_completion.choices[0].message.content

        # 将助手回复添加到消息列表，以支持多轮对话
        assistant_message = {
            "role": "assistant",
            "content": assistant_reply
        }
        st.session_state.messages.append(assistant_message)
        return assistant_reply

    except Exception as e:
        print(f"An error occurred: {e}")

# 修正后的DeepSeek API调用函数
def call_deepseek_api(prompt, client):
    try:
        # 获取用户输入
        user_input = prompt
        # 调用 API 进行流式对话
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=st.session_state.messages,
            stream=True
        )
        return response
    except Exception as e:
        return f"DeepSeek API请求发生错误：{str(e)}"

# SiliconCloud API调用函数
def call_silicon_api(prompt, client):
    try:
        # 获取用户输入
        user_input = prompt
        # 调用 API 进行流式对话
        response = client.chat.completions.create(
            model=silicon_model_name,
            messages=st.session_state.messages,
            stream=True
        )
        return response
    except Exception as e:
        return f"SiliconCloud API请求发生错误：{str(e)}"
# 主函数
def main():
    # 主界面
    st.title(f"✨ AI Chat 🤗 {selected_model}")


    # OCR处理
    if uploaded_file:
        with st.spinner("正在解析文件..."):
            st.session_state.ocr_text = process_file(uploaded_file)
        st.text_area("解析结果", st.session_state.ocr_text, height=150)

    # 创建一个按钮，点击后触发文件上传

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入处理
    if prompt := st.chat_input("输入消息..."):
        final_prompt = st.session_state.ocr_text if st.session_state.ocr_text else prompt
        
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with st.chat_message("user"):
            st.markdown(final_prompt)
        
        # 准备生成回复
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # 根据选择的模型调用不同的API
            if selected_model == "Ollama":
                for chunk in stream_ollama_response(final_prompt, ollama_model_name):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

            elif selected_model == "Groq":
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                for chunk in call_groq_api(final_prompt, client):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

            elif selected_model == "DeepSeek":
                client = OpenAI(api_key=silicon_api_key, base_url=silicon_url)
                try:
                    # 推理
                    response = call_deepseek_api(final_prompt, client)
                    full_response += "<think>"
                    for chunk in response:
                        delta = chunk.choices[0].delta
                        if hasattr(delta,'reasoning_content') and delta.reasoning_content is not None:
                            reasoning_chunk = delta.reasoning_content
                            response_placeholder.markdown(full_response + "▌")
                            full_response += reasoning_chunk
                        else:
                            break
                    full_response += r"<\think>"
                    if hasattr(delta, 'content') and delta.content is not None:
                        content_chunk = delta.content
                        full_response += content_chunk
                        response_placeholder.markdown(full_response + "▌")
                    for chunk in response:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content is not None:
                            content_chunk = delta.content
                            response_placeholder.markdown(full_response + "▌")
                            full_response += content_chunk
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"发生错误: {e}, 可能是 DeepSeek API 繁忙，请稍后再试。")
            
            elif selected_model == "SiliconCloud":
                client = OpenAI(api_key=silicon_api_key, base_url=silicon_url)
                try:
                    # 推理
                    response = call_silicon_api(final_prompt, client)
                    full_response += "<think>"
                    for chunk in response:
                        delta = chunk.choices[0].delta
                        if hasattr(delta,'reasoning_content') and delta.reasoning_content is not None:
                            reasoning_chunk = delta.reasoning_content
                            response_placeholder.markdown(full_response + "▌")
                            full_response += reasoning_chunk
                        else:
                            break
                    full_response += r"<\think>"
                    if hasattr(delta, 'content') and delta.content is not None:
                        content_chunk = delta.content
                        full_response += content_chunk
                        response_placeholder.markdown(full_response + "▌")
                    for chunk in response:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content is not None:
                            content_chunk = delta.content
                            response_placeholder.markdown(full_response + "▌")
                            full_response += content_chunk
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"发生错误: {e}, 可能是 Silicon API 繁忙，请稍后再试。")
            
            # 最终更新去掉光标
            response_placeholder.markdown(full_response)
        
        # 添加到消息历史
        if selected_model == "Ollama":
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
