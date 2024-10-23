import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载对话模型 "meta-llama/Llama-3.2-1B"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 或您选择的其他模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Load model directly
# from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
# processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
# tokenizer = AutoTokenizer.from_pretrained("openai/whisper-large-v3-turbo")
# model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")

# 设置 Streamlit 页面
st.title("道生中医养生堂")
st.write("与模型对话：")

# 初始化聊天记录
if 'history' not in st.session_state:
    st.session_state.history = []

# 用户输入
user_input = st.text_input("您：", "")

# 按下发送按钮
if st.button("发送"):
    # 添加用户输入到历史记录
    st.session_state.history.append({"role": "user", "content": user_input})

    # 准备输入
    input_text = ""
    for chat in st.session_state.history:
        input_text += f"{chat['role']}: {chat['content']}\n"

    # 生成模型的回复
    inputs = tokenizer.encode(input_text + "assistant:", return_tensors="pt")
    outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # 解码模型回复
    response = tokenizer.decode(outputs[:, inputs.shape[1]:][0], skip_special_tokens=True)

    # 添加模型回复到历史记录
    st.session_state.history.append({"role": "assistant", "content": response})

# 显示聊天记录
for chat in st.session_state.history:
    if chat["role"] == "user":
        st.write(f"**您**: {chat['content']}")
    else:
        st.write(f"**道生AI**: {chat['content']}")
