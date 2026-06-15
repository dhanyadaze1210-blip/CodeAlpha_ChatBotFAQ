import streamlit as st
import nltk
import os

nltk_data_path = "/tmp/nltk_data"
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

nltk.download('punkt', quiet=True)

nltk.download('stopwords', quiet=True)

st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI FAQ Chatbot")
st.caption("Your personal AI knowledge assistant — ask me anything about Artificial Intelligence")
st.divider()

faq = {
    "What is AI?":
        "Artificial Intelligence is technology that enables machines to perform tasks that normally require human intelligence, such as understanding language, recognising images, and making decisions.",
    "What is Machine Learning?":
        "Machine Learning is a subset of AI where computers learn patterns from data without being explicitly programmed. The more data it sees, the better it gets.",
    "What is Deep Learning?":
        "Deep Learning is a subset of Machine Learning that uses neural networks with many layers. It powers things like face recognition, speech assistants, and self-driving cars.",
    "What is a Neural Network?":
        "A Neural Network is a system inspired by the human brain. It consists of layers of nodes that process and pass information to find patterns in data.",
    "What is NLP?":
        "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language. It powers chatbots, translators, and voice assistants.",
    "What is Generative AI?":
        "Generative AI is a type of AI that creates new content such as text, images, music, and code. Examples include ChatGPT and DALL-E.",
    "What is a Large Language Model?":
        "A Large Language Model (LLM) is a deep learning model trained on massive amounts of text. It can generate, summarise, and answer questions in natural language. GPT-4 is an example.",
    "What is Computer Vision?":
        "Computer Vision is a field of AI that enables computers to interpret and understand visual information from images and videos, like recognising faces or detecting objects.",
    "What is Reinforcement Learning?":
        "Reinforcement Learning is a type of Machine Learning where an agent learns by trial and error, receiving rewards for good actions and penalties for bad ones.",
    "What is Data Science?":
        "Data Science is the field of extracting insights and knowledge from structured and unstructured data using statistics, programming, and machine learning.",
    "What is an Algorithm?":
        "An algorithm is a step-by-step set of instructions given to a computer to solve a problem or complete a task.",
    "What is Python used for in AI?":
        "Python is the most popular language for AI because it has powerful libraries like NumPy, Pandas, TensorFlow, PyTorch, and Scikit-learn that make building AI models much easier.",
    "What is TensorFlow?":
        "TensorFlow is an open-source deep learning framework developed by Google. It is used to build and train neural network models.",
    "What is overfitting?":
        "Overfitting happens when a model learns the training data too well, including its noise, and performs poorly on new unseen data.",
    "What is the difference between AI and ML?":
        "AI is the broad concept of machines being intelligent. ML is a specific method to achieve AI by training machines on data. All ML is AI but not all AI is ML.",
}

faq_aliases = {
    "What is AI?": ["artificial intelligence", "what is artificial intelligence", "tell me about ai", "explain ai"],
    "What is Machine Learning?": ["ml", "machine learning basics", "how does ml work"],
    "What is Deep Learning?": ["dl", "deep learning basics"],
    "What is NLP?": ["natural language processing", "nlp basics"],
    "What is Generative AI?": ["generative ai", "chatgpt", "what is chatgpt", "dall-e"],
    "What is a Large Language Model?": ["llm", "large language model", "gpt", "gpt-4"],
}

def clean(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)
faq_aliases = {
    "What is AI?": ["artificial intelligence", "what is artificial intelligence", "tell me about ai", "explain ai"],
    "What is Machine Learning?": ["ml", "machine learning basics", "how does ml work"],
    "What is Deep Learning?": ["dl", "deep learning basics"],
    "What is NLP?": ["natural language processing", "nlp basics"],
    "What is Generative AI?": ["generative ai", "chatgpt", "what is chatgpt", "dall-e"],
    "What is a Large Language Model?": ["llm", "large language model", "gpt", "gpt-4"],
}
questions = list(faq.keys())
answers   = list(faq.values())
cleaned_questions = [clean(q) for q in questions]

def get_answer(user_input):
    cleaned_input = clean(user_input)
    
    # Check aliases first
    for question, aliases in faq_aliases.items():
        for alias in aliases:
            if alias in user_input.lower():
                index = questions.index(question)
                return index, 1.0
    
    # Fall back to TF-IDF cosine similarity
    all_texts = cleaned_questions + [cleaned_input]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_index = similarity.argmax()
    best_score = similarity[0][best_index]
    return best_index, best_score

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("💡 Suggested Questions")
    st.write("Click any topic to copy it:")

    suggested = [
        "What is AI?",
        "What is Machine Learning?",
        "What is Deep Learning?",
        "What is NLP?",
        "What is Generative AI?",
        "What is a Neural Network?",
        "What is Python used for in AI?",
        "What is the difference between AI and ML?",
    ]

    for s in suggested:
        st.code(s, language=None)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("15 AI topics loaded")
    st.caption("Built by Dhanya R")
    st.caption("CodeAlpha Internship 2026")

# ── Stats row ─────────────────────────────────────────────
if "chat_history" in st.session_state and st.session_state.chat_history:
    total = len(st.session_state.chat_history)
    matched = sum(1 for c in st.session_state.chat_history if c["score"] >= 0.1)
    avg_score = sum(c["score"] for c in st.session_state.chat_history) / total

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💬 Questions Asked", total)
    with col2:
        st.metric("✅ Answered", matched)
    with col3:
        st.metric("🎯 Avg Confidence", f"{avg_score:.0%}")

    st.divider()

# ── Chat history ──────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not st.session_state.chat_history:
    st.info("👋 Hello! I am your AI FAQ assistant. Type any question about Artificial Intelligence below and I will do my best to answer it!")

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"📌 Matched: *{chat['matched']}*")
        with col_b:
            if chat["score"] >= 0.6:
                st.caption(f"🎯 Confidence: {chat['score']:.0%} 🟢")
            elif chat["score"] >= 0.3:
                st.caption(f"🎯 Confidence: {chat['score']:.0%} 🟡")
            else:
                st.caption(f"🎯 Confidence: {chat['score']:.0%} 🔴")

# ── Chat input ────────────────────────────────────────────
user_input = st.chat_input("Ask me about AI, ML, Deep Learning, NLP...")

if user_input:
    index, score = get_answer(user_input)

    if score < 0.1:
        answer  = "I am sorry, I do not have an answer for that. Try asking about AI, Machine Learning, Deep Learning, NLP, Computer Vision, or Python!"
        matched = "No match found"
    else:
        answer  = answers[index]
        matched = questions[index]

    st.session_state.chat_history.append({
        "question": user_input,
        "answer"  : answer,
        "matched" : matched,
        "score"   : score,
    })

    st.rerun()

st.divider()
st.caption("🤖 AI FAQ Chatbot  ·  Built with Python, NLTK & Scikit-learn  ·  CodeAlpha AI Internship 2026  ·  Dhanya R")
