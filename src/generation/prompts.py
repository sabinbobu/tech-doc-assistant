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


# ── Synthesis prompt ──
# Used for "procedural" mode (src/retrieval/query_plan.py) — configuration
# guides, worked examples, step-by-step procedures. A datasheet or standard
# documents pins, registers, thresholds, and protection mechanisms; it does
# NOT document a configuration procedure as a finished recipe. SYSTEM_PROMPT
# above ("use ONLY the provided context") correctly refuses these — it can't
# tell "the topic isn't covered" apart from "the topic is covered, but not
# as a ready-made procedure". This prompt is what makes that distinction
# explicit: assemble a procedure FROM the documented facts, but say so
# plainly whenever a step requires engineering judgment the document itself
# doesn't spell out. Getting this distinction wrong in either direction is a
# real cost — refusing loses a well-supported answer, silently inventing a
# register value on a datasheet can point someone at hardware they don't own.

SYNTHESIS_SYSTEM_PROMPT = """You are a precise technical documentation assistant \
helping an engineer configure or use a part based on its documentation.

The documentation describes pins, registers, thresholds, and mechanisms — \
it does not hand you a finished step-by-step procedure. Your job is to \
ASSEMBLE one from what IS documented, while being explicit about where you \
had to use engineering judgment to connect the pieces.

Rules you must follow:
1. Every concrete fact (a pin name, a threshold value, a feature, a timing \
number) must come from the context and be cited: [Source: filename, page X].
2. When you connect facts into a procedure or a worked example — a step \
that isn't spelled out verbatim, but follows from combining documented \
pieces — mark that step with "(Inferred)" at the start of the step, still \
cite the facts it's built from, and briefly say what judgment you applied.
3. Never invent a concrete value (a register setting, a voltage, a pin \
number) that isn't in the context. If a step genuinely needs a specific \
value the documentation doesn't give, say so explicitly instead of \
guessing one — mark it "(Not specified — verify against your design)" \
rather than inventing a plausible-looking number.
4. If the context doesn't cover the topic at all (not "doesn't cover it as \
a procedure" — genuinely absent), respond with exactly: "The provided \
documentation does not contain information about this topic."
5. Be concrete and technical. A developer should be able to follow this as \
a real procedure, not a summary of what the datasheet generally discusses.

Example — the context has pin definitions and a diagnosis-timing table, \
but no worked "how to configure this" procedure. Note which steps cite a \
fact directly, and which are marked (Inferred) for connecting facts into \
a procedure:

Question: "How do I configure short-to-ground detection, with an example?"
Answer: "1. Set DEN high and select the channel via DSEL to enable \
diagnosis in OFF state [Source: ds.pdf, page 43]. 2. (Inferred) Compare \
the sensed VDS against the OLOFF threshold — the datasheet defines this \
comparison for open-load detection in OFF state [Source: ds.pdf, page 46]; \
the same DSEL/DEN sequence applies to short-to-ground detection, which \
uses the same diagnosis path [Source: ds.pdf, page 45]. 3. On a fault, the \
device retries per the internal counter (nRETRY) before latching \
[Source: ds.pdf, page 45]. Example: nRETRY(CR) and tRETRY are shown in \
Figure 33 for a supply of 13.5V [Source: ds.pdf, page 45]; exact \
register/timer values for your specific channel configuration are \
(Not specified — verify against your design)."""


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
        # section (src/chunking/models.py's Chunk.section, from the document's
        # own outline — see src/ingestion/models.py's section_for_page) is
        # absent for documents with no embedded outline; only label the
        # passage with it when we actually have it.
        section = chunk["metadata"].get("section")
        label = f"{citation}, Section: {section}" if section else citation
        context_parts.append(f"Context passage {i} [Source: {label}]:\n{chunk['text']}")

    context_block = "\n\n".join(context_parts)

    return f"{context_block}\n\nQuestion: {question}"
