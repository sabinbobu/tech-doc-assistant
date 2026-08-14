"""
Prompt templates for RAG answer generation.

WHY PROMPTS ARE IN THEIR OWN MODULE:
Prompts are logic, not strings. They determine answer quality,
citation format, and how the LLM handles edge cases.
Keeping them here means you can iterate on prompts independently
of the generation code — and test them in isolation.

Think of prompts like PID tuning parameters in your embedded world:
small changes have large effects, and you want them versioned and
auditable, not buried in business logic.
"""

# ── System prompt ──
# Sets the LLM's role and hard constraints for every conversation.
# The key constraints:
#   1. Only use provided context — no general knowledge
#   2. Always cite sources with page numbers
#   3. Explicitly say when the answer isn't in the context
#      (this is critical — an honest "I don't know" beats a hallucination)

SYSTEM_PROMPT = """You are a precise technical documentation assistant.

Your job is to answer questions based EXCLUSIVELY on the context passages provided.

Rules you must follow:
1. Use ONLY information from the provided context. Never use general knowledge.
2. After each claim, cite the source in this format: [Source: filename, page X]
3. If the context only partially answers the question, answer with what IS
   supported by the context, cite it, and explicitly note what is missing.
   This applies to multi-part questions too: if the question has several
   parts (e.g. "what X does it have, and how do I do Y with it"), answer
   each part the context supports and separately note which parts it
   doesn't cover — never let one unsupported part cause you to refuse the
   whole question. Only respond with exactly: "The provided documentation
   does not contain information about this topic." if NONE of the provided
   context passages are relevant to any part of the question.
4. Be concise and technical. Do not pad your answer with filler phrases.
5. If multiple sources support a claim, cite all of them.

Example of rule 3 (a multi-part question where the context covers one part
but not the other — note that the answer covers the supported part in full,
and only flags the unsupported part, instead of refusing everything):

Question: "What fault types does the sensor report, and how do I calibrate it in Python?"
Answer: "The sensor reports Overtemperature, Overvoltage, and Short-Circuit
faults [Source: manual.pdf, page 12]. The documentation does not cover
Python calibration code — it only describes the hardware calibration
procedure using the CAL pin [Source: manual.pdf, page 15].\""""


def build_user_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Build the user-turn prompt by injecting retrieved chunks into a template.

    This is called "prompt construction" or "context stuffing" —
    you're literally inserting the retrieved evidence into the prompt
    so the LLM has it available when generating the answer.

    The format we use:
        Context passage 1 [Source: file.pdf, page N]:
        <text>

        Context passage 2 [Source: file.pdf, page N]:
        <text>

        Question: <user question>

    Args:
        question: The user's original question.
        context_chunks: List of dicts from query_collection() —
                        each has 'text' and 'metadata' keys.

    Returns:
        Fully constructed prompt string ready to send to the LLM.
    """
    # Build the context block — each chunk labeled with its source
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        citation = chunk["metadata"].get("citation", "Unknown source")
        context_parts.append(
            f"Context passage {i} [Source: {citation}]:\n{chunk['text']}"
        )

    context_block = "\n\n".join(context_parts)

    return f"{context_block}\n\nQuestion: {question}"
