"""
ui/tabs/copilot.py — LLM-powered analyst grounded in the user's own data.

Renders the Copilot chat panel. All conversation state lives in
st.session_state so it survives reruns. Tool execution is delegated to
core.copilot.ask_copilot which handles the Anthropic tool-use loop.
"""

from __future__ import annotations

import os
import streamlit as st


def render(db_path: str) -> None:
    """Render the Copilot tab.

    Args:
      db_path: path to the screener SQLite DB, forwarded to ask_copilot
               so its tools can query the latest screener rows.
    """
    st.subheader("🤖 TradeStrategy Copilot")
    st.caption(
        "Ask questions about any ticker, your screener results, or your "
        "open trades. The Copilot uses Claude with direct access to your "
        "screener DB, recommendation engine, peer fundamentals, and "
        "catalyst data — every number it cites comes from a tool call, "
        "not the model's training data."
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning(
            "**ANTHROPIC_API_KEY not set.** Add it to your shell environment "
            "or to `TradeStrategy/.env`:\n\n"
            "```\nANTHROPIC_API_KEY=sk-ant-...\n```\n\n"
            "Then restart the Streamlit app. Get a key at "
            "[console.anthropic.com](https://console.anthropic.com/)."
        )

    if "copilot_history" not in st.session_state:
        st.session_state.copilot_history = []   # API-format messages (for sending)
    if "copilot_display" not in st.session_state:
        st.session_state.copilot_display = []   # plain text turns (for rendering)

    def _safe_md(text: str) -> str:
        # Streamlit's markdown treats `$…$` as LaTeX, which mangles dollar amounts
        # in LLM output ("$2,281" disappears, apostrophes become primes).
        return text.replace("$", "\\$")

    # Replay prior turns
    for turn in st.session_state.copilot_display:
        with st.chat_message(turn["role"]):
            st.markdown(_safe_md(turn["content"]))

    # Suggested prompt chips on first load
    suggestion = None
    if not st.session_state.copilot_display:
        st.markdown("**Try asking:**")
        c1, c2, c3 = st.columns(3)
        if c1.button("Is MSFT a good buy right now?", width='stretch'):
            suggestion = "Is MSFT a good buy right now? Give me a full read with peer comparison and catalyst risk."
        if c2.button("What are today's top 5 long setups?", width='stretch'):
            suggestion = "What are today's top 5 long setups? For each, tell me what makes it interesting and what the main risk is."
        if c3.button("Review my open trades", width='stretch'):
            suggestion = "Look at my open trades — anything I should be paying attention to?"

    user_input = st.chat_input(
        "Ask anything — e.g. 'Is NVDA overextended?' or 'Compare AAPL vs MSFT'",
        disabled=not api_key,
    )
    prompt = user_input or suggestion

    if prompt and api_key:
        # Render the user turn immediately
        with st.chat_message("user"):
            st.markdown(_safe_md(prompt))
        st.session_state.copilot_display.append({"role": "user", "content": prompt})
        st.session_state.copilot_history.append({"role": "user", "content": prompt})

        # Run the tool loop
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    from core.copilot import ask_copilot
                    final_text, updated_history = ask_copilot(
                        st.session_state.copilot_history,
                        db_path=db_path,
                        api_key=api_key,
                    )
                except Exception as e:
                    final_text = f"⚠️ Copilot error: `{type(e).__name__}: {e}`"
                    updated_history = st.session_state.copilot_history
            st.markdown(_safe_md(final_text))

        st.session_state.copilot_history = updated_history
        st.session_state.copilot_display.append({"role": "assistant", "content": final_text})
        st.rerun()

    if st.session_state.copilot_display:
        if st.button("🧹 Clear conversation", key="copilot_clear"):
            st.session_state.copilot_history = []
            st.session_state.copilot_display = []
            st.rerun()
