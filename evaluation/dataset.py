"""
Ground truth evaluation dataset for the Technical Documentation Assistant.

WHY WE HAND-CRAFT THIS:
Automated datasets have subtle errors. For a portfolio project,
hand-crafted questions signal that you actually understand the domain
and took evaluation seriously — not just ran a script.

Each entry has:
- question:      what a real user would ask
- ground_truth:  the correct answer based on the actual document
- category:      type of question (definition, process, classification)
                 useful for analyzing WHERE your system fails

HOW TO EXTEND THIS:
Read your MISRA PDF, find important concepts, write questions as if
you're a developer who just joined a project using MISRA.
Aim for 20+ questions in a real evaluation suite.
"""

EVALUATION_DATASET = [
    {
        "question": "What is a deviation in MISRA compliance?",
        "ground_truth": (
            "A deviation is a formal process that permits a project to use a guideline "
            "in a manner different from that specified. Deviations must be documented "
            "and authorized before the software is released."
        ),
        "category": "definition",
    },
    {
        "question": "What are the two categories of MISRA guidelines?",
        "ground_truth": (
            "MISRA guidelines are categorized as either Mandatory or Advisory. "
            "Mandatory guidelines must always be followed — violations cannot be permitted. "
            "Advisory guidelines are recommended but may be deviated from with justification."
        ),
        "category": "classification",
    },
    {
        "question": "What is a violation in the context of MISRA compliance?",
        "ground_truth": (
            "A violation occurs when source code does not comply with a MISRA guideline. "
            "Violations of Mandatory guidelines cannot be permitted under any circumstances. "
            "Violations of Advisory guidelines may be permitted through the deviation process."
        ),
        "category": "definition",
    },
    {
        "question": "What is required to claim MISRA compliance for a software component?",
        "ground_truth": (
            "To claim MISRA compliance, all Mandatory guidelines must be followed with zero "
            "violations. Any deviations from Advisory guidelines must be formally documented "
            "and authorized. A compliance summary should record the guidelines checked, "
            "tools used, and any deviations or violations."
        ),
        "category": "process",
    },
    {
        "question": "What is a permit in MISRA compliance?",
        "ground_truth": (
            "A permit is an authorization that allows a specific deviation from a guideline "
            "within a defined scope. Permits must be documented and approved, and they apply "
            "only to the specific context for which they were granted."
        ),
        "category": "definition",
    },
    {
        "question": "Can a mandatory MISRA guideline be deviated from?",
        "ground_truth": (
            "No. Mandatory guidelines shall always be complied with. There is no deviation "
            "process for Mandatory guidelines — they represent absolute requirements that "
            "cannot be relaxed under any circumstances."
        ),
        "category": "process",
    },
    {
        "question": "What should a MISRA compliance summary document contain?",
        "ground_truth": (
            "A compliance summary should document the scope of the compliance claim, "
            "which guidelines were checked, what analysis tools were used, any deviations "
            "that were granted, and any violations that were found and their resolution."
        ),
        "category": "process",
    },
    {
        "question": "What is the difference between a guideline and a rule in MISRA?",
        "ground_truth": (
            "In MISRA, a guideline is the general term for any MISRA requirement, "
            "which includes both rules and directives. Rules are specific, statically "
            "checkable requirements. Directives are requirements where full compliance "
            "checking may require additional information beyond the source code alone."
        ),
        "category": "definition",
    },
]
