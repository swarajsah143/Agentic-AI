from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv
from google.genai.errors import APIError
from langchain_google_genai import ChatGoogleGenerativeAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "blood_report.txt"

load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="Gemini Notebook Studio",
    page_icon="✨",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1080px; padding-top: 2.5rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128, 128, 128, 0.18);}
    .eyebrow {color: #6c63ff; font-size: 0.78rem; font-weight: 700;
              letter-spacing: 0.12em; text-transform: uppercase;}
    </style>
    """,
    unsafe_allow_html=True,
)


def response_to_text(content: object) -> str:
    """Convert LangChain's text or block-based message content to displayable text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        if text_parts:
            return "\n".join(text_parts)
    return str(content)


def invoke_gemini(prompt: str, model_name: str, temperature: float) -> str:
    response = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
    ).invoke(prompt)
    return response_to_text(response.content)


try:
    blood_report = REPORT_PATH.read_text(encoding="utf-8")
    report_error = None
except OSError as error:
    blood_report = None
    report_error = str(error)

with st.sidebar:
    st.markdown("## ✨ Notebook settings")
    st.caption("Model and sampling controls from `llm_calling.ipynb`")
    model_name = st.text_input("Gemini model", value="gemini-3.5-flash").strip()
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=1.6,
        step=0.1,
        help="The notebook uses temperature 0 for its first example and 1.6 afterward.",
    )
    st.divider()
    st.caption("Add `GOOGLE_API_KEY` to the project `.env` file to enable Gemini.")

st.markdown('<p class="eyebrow">LangChain · Google Gemini</p>', unsafe_allow_html=True)
st.title("Notebook workflows")
st.write(
    "Run the notebook's blood-report extraction and diet-plan stages, or try a "
    "standalone prompt with adjustable temperature."
)

analysis_tab, prompt_tab, report_tab = st.tabs(
    ["🧪 Blood report workflow", "💬 Prompt playground", "📄 Local blood report"]
)

with analysis_tab:
    st.markdown("### Stage 1 · Extract and classify results")
    st.write(
        "Extract each test value and classify it as HIGH, LOW, or NORMAL using "
        "the reference range printed in the report."
    )
    st.warning(
        "This sends `blood_report.txt` to Google Gemini when you run Stage 1. "
        "The generated analysis and diet suggestions are informational only, "
        "not medical advice."
    )

    extract_clicked = st.button(
        "Extract and classify results",
        type="primary",
        disabled=blood_report is None,
        key="extract_report",
    )
    if blood_report is None:
        st.error(f"Could not read the blood report: {report_error}")
    elif extract_clicked:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error(
                "Gemini is not configured. Add `GOOGLE_API_KEY` to the project `.env` "
                "file, then restart the app."
            )
        elif not model_name:
            st.error("Enter a Gemini model name in the sidebar.")
        else:
            extraction_prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract ALL test values and classify each one as HIGH, LOW, or NORMAL
based on the reference ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{blood_report}
"""
            try:
                with st.spinner("Extracting and classifying test results..."):
                    extracted_values = invoke_gemini(
                        extraction_prompt,
                        model_name,
                        temperature,
                    )
            except APIError as error:
                st.error(f"Gemini could not complete Stage 1: {error}")
            else:
                st.session_state["extracted_values"] = extracted_values
                st.session_state["analysis_model"] = model_name
                st.session_state["analysis_temperature"] = temperature
                st.session_state.pop("diet_response", None)

    if "extracted_values" in st.session_state:
        st.markdown("#### Extracted values")
        st.caption(
            f"{st.session_state['analysis_model']} · "
            f"temperature {st.session_state['analysis_temperature']:.1f}"
        )
        st.markdown(st.session_state["extracted_values"])

        st.divider()
        st.markdown("### Stage 2 · Health summary and Indian diet plan")
        st.write(
            "Generate a brief plain-language summary and the two diet-plan sections "
            "requested in the notebook."
        )
        diet_clicked = st.button(
            "Generate health summary and diet plan",
            type="primary",
            key="generate_diet",
        )
        if diet_clicked:
            diet_prompt = f"""
You are a clinical nutritionist specializing in Indian dietary habits.

Based on the blood work analysis below, write:
1. A short health summary in 4-5 lines explaining the patient's condition in simple language
2. A short, practical Indian diet plan having only two sections (1) Foods to avoid (2) Foods to eat more of.
   Do not include any other sections in diet plan.

Blood Work Analysis:
{st.session_state["extracted_values"]}
"""
            try:
                with st.spinner("Generating the summary and diet plan..."):
                    diet_response = invoke_gemini(
                        diet_prompt,
                        st.session_state["analysis_model"],
                        st.session_state["analysis_temperature"],
                    )
            except APIError as error:
                st.error(f"Gemini could not complete Stage 2: {error}")
            else:
                st.session_state["diet_response"] = diet_response

    if "diet_response" in st.session_state:
        st.markdown("#### Health summary and diet plan")
        st.caption(
            f"Using the same model and temperature as Stage 1: "
            f"{st.session_state['analysis_model']} · "
            f"{st.session_state['analysis_temperature']:.1f}"
        )
        st.markdown(st.session_state["diet_response"])

with prompt_tab:
    st.markdown("### Standalone Gemini prompt")
    st.write("This recreates the notebook's initial prompt-response examples.")
    with st.form("prompt_form"):
        prompt = st.text_area(
            "Your prompt",
            value="Who is Elon Musk? Define him in 2 simple sentences.",
            height=130,
            placeholder="Ask Gemini something...",
        )
        submitted = st.form_submit_button(
            "Generate response",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not prompt.strip():
            st.warning("Enter a prompt before generating a response.")
        elif not os.getenv("GOOGLE_API_KEY"):
            st.error(
                "Gemini is not configured. Add `GOOGLE_API_KEY` to the project `.env` "
                "file, then restart the app."
            )
        elif not model_name:
            st.error("Enter a Gemini model name in the sidebar.")
        else:
            try:
                with st.spinner("Waiting for Gemini..."):
                    prompt_response = invoke_gemini(
                        prompt.strip(),
                        model_name,
                        temperature,
                    )
            except APIError as error:
                st.error(f"Gemini could not complete the request: {error}")
            else:
                st.session_state["prompt_response"] = prompt_response
                st.session_state["prompt_model"] = model_name
                st.session_state["prompt_temperature"] = temperature

    if "prompt_response" in st.session_state:
        st.markdown("#### Gemini response")
        st.caption(
            f"{st.session_state['prompt_model']} · "
            f"temperature {st.session_state['prompt_temperature']:.1f}"
        )
        st.markdown(st.session_state["prompt_response"])

with report_tab:
    st.markdown("### `blood_report.txt`")
    st.caption("Read from the project folder. This preview stays in the app.")
    if blood_report is None:
        st.error(f"Could not read the blood report: {report_error}")
    else:
        st.text(blood_report)
