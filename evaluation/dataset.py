"""
Ground truth evaluation dataset for the Technical Documentation Assistant.

WHY WE HAND-CRAFT THIS:
Automated datasets have subtle errors. For a portfolio project,
hand-crafted questions signal that you actually understand the domain
and took evaluation seriously — not just ran a script.

Each entry has:
- question:         what a real user would ask
- ground_truth:      the correct answer based on the actual document
- category:          type of question (definition, process, classification)
                     useful for analyzing WHERE your system fails
- gold_doc:          the source PDF filename that actually answers this question
- gold_pages:        the 1-indexed page number(s) within gold_doc that contain
                     the answer. Read directly from the source PDF, not
                     guessed — every entry below was verified against the
                     actual extracted page text before being recorded.
                     Lets evaluation/retrieval_metrics.py compute recall@k,
                     MRR, and nDCG deterministically (did retrieval actually
                     surface the right page?) instead of relying only on
                     RAGAS's slower, LLM-judged, non-deterministic scoring.
- identifier_heavy:  True if the question's answer hinges on an exact MISRA
                     term or spec value (a category name like "Mandatory",
                     a defined term like "GEP", a spec number) that dense
                     embedding search can blur but BM25's exact-term
                     matching should catch. Lets Phase 2's compare.py check
                     whether hybrid search is actually helping where theory
                     predicts it should, not just in aggregate.

HOW TO EXTEND THIS:
Read the PDF, find important concepts, write questions as if you're a
developer who just joined a project using MISRA (or a customer with this
Epson printer). Verify gold_pages against the actual document — don't guess.
"""

EVALUATION_DATASET = [
    # ── Original 8 (MISRA-Compliance-2020.pdf) — gold_pages/gold_doc/
    #    identifier_heavy added retroactively; question/ground_truth/category
    #    are unchanged from the original hand-written set.
    {
        "question": "What is a deviation in MISRA compliance?",
        "ground_truth": (
            "A deviation is a formal process that permits a project to use a guideline "
            "in a manner different from that specified. Deviations must be documented "
            "and authorized before the software is released."
        ),
        "category": "definition",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [20, 38],
        "identifier_heavy": False,
    },
    {
        "question": "What are the two categories of MISRA guidelines?",
        "ground_truth": (
            "MISRA guidelines are categorized as either Mandatory or Advisory. "
            "Mandatory guidelines must always be followed — violations cannot be permitted. "
            "Advisory guidelines are recommended but may be deviated from with justification."
        ),
        "category": "classification",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [23],
        "identifier_heavy": True,
    },
    {
        "question": "What is a violation in the context of MISRA compliance?",
        "ground_truth": (
            "A violation occurs when source code does not comply with a MISRA guideline. "
            "Violations of Mandatory guidelines cannot be permitted under any circumstances. "
            "Violations of Advisory guidelines may be permitted through the deviation process."
        ),
        "category": "definition",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [20, 39],
        "identifier_heavy": False,
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
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [28, 31],
        "identifier_heavy": False,
    },
    {
        "question": "What is a permit in MISRA compliance?",
        "ground_truth": (
            "A permit is an authorization that allows a specific deviation from a guideline "
            "within a defined scope. Permits must be documented and approved, and they apply "
            "only to the specific context for which they were granted."
        ),
        "category": "definition",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [20, 21, 38],
        "identifier_heavy": False,
    },
    {
        "question": "Can a mandatory MISRA guideline be deviated from?",
        "ground_truth": (
            "No. Mandatory guidelines shall always be complied with. There is no deviation "
            "process for Mandatory guidelines — they represent absolute requirements that "
            "cannot be relaxed under any circumstances."
        ),
        "category": "process",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [23],
        "identifier_heavy": True,
    },
    {
        "question": "What should a MISRA compliance summary document contain?",
        "ground_truth": (
            "A compliance summary should document the scope of the compliance claim, "
            "which guidelines were checked, what analysis tools were used, any deviations "
            "that were granted, and any violations that were found and their resolution."
        ),
        "category": "process",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [29, 30],
        "identifier_heavy": False,
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
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [16, 38, 39],
        "identifier_heavy": True,
    },
    # ── New MISRA-Compliance-2020.pdf questions ──
    {
        "question": "What is a guideline enforcement plan (GEP)?",
        "ground_truth": (
            "A guideline enforcement plan (GEP) lists each guideline within The Guidelines "
            "and is produced to indicate how compliance with the guidelines is to be "
            "checked. The supplier makes the GEP available to the acquirer so the "
            "suitability and robustness of the checking undertaken can be assessed."
        ),
        "category": "definition",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [16],
        "identifier_heavy": True,
    },
    {
        "question": "What are the two types of analysis scope for a MISRA rule?",
        "ground_truth": (
            "The analysis scope of each MISRA rule is described as either 'Single "
            "Translation Unit' or 'System'. Many rules can be checked by examining each "
            "translation unit in isolation; some can only be fully checked by analysing "
            "the entire system's source code."
        ),
        "category": "classification",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [16],
        "identifier_heavy": True,
    },
    {
        "question": "What does it mean for a MISRA rule to be undecidable?",
        "ground_truth": (
            "When a rule is undecidable, no analysis tool, however sophisticated, can "
            "guarantee to respond unequivocally to whether code complies with it in "
            "every situation. A tool may report 'Yes', 'No', or in many cases only "
            "'Possibly'."
        ),
        "category": "definition",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [19],
        "identifier_heavy": False,
    },
    {
        "question": "What information should a deviation record include?",
        "ground_truth": (
            "A deviation record should include: the guideline(s) being violated; a "
            "concise description of the circumstances in which the violation is "
            "acceptable; the reason the deviation is required; background information "
            "explaining the context and language issues; and a set of requirements "
            "including risk assessment procedures and precautions to be observed."
        ),
        "category": "process",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [20],
        "identifier_heavy": False,
    },
    {
        "question": "What are the three sources from which deviation permits may originate?",
        "ground_truth": (
            "Deviation permits may originate from three sources: public deviation "
            "permits published by MISRA, acquirer deviation permits produced by an "
            "acquirer, and supplier deviation permits produced by a supplier."
        ),
        "category": "classification",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [21],
        "identifier_heavy": False,
    },
    {
        "question": "What are the acceptable reasons for justifying a MISRA deviation?",
        "ground_truth": (
            "A deviation may be justified for one of four reasons: Code quality (the "
            "guideline compromises another aspect of software quality), Access to "
            "hardware (compiler-specific extensions needed for low-level hardware "
            "access), Adopted code integration (translation units compliant in "
            "isolation become non-compliant when combined), and Non-compliant adopted "
            "code (adopted code was never developed with MISRA compliance as an "
            "objective)."
        ),
        "category": "classification",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [21, 22],
        "identifier_heavy": False,
    },
    {
        "question": "Can a Mandatory guideline be re-categorized to a different category?",
        "ground_truth": (
            "No. A Mandatory guideline may not be re-categorized in any way. A Required "
            "guideline may not be re-categorized as Advisory or Disapplied, but may be "
            "re-categorized as Mandatory."
        ),
        "category": "process",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [23],
        "identifier_heavy": True,
    },
    {
        "question": (
            "What happens to violations of a guideline that has been re-categorized as "
            "Disapplied?"
        ),
        "ground_truth": (
            "Violations of guidelines which have been re-categorized as Disapplied are "
            "disregarded altogether — they do not need to be identified or supported by "
            "a deviation."
        ),
        "category": "definition",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [23, 24],
        "identifier_heavy": True,
    },
    {
        "question": "What is adopted code, and how does it differ from native code?",
        "ground_truth": (
            "Adopted code is code derived from outside the scope of the current project "
            "(e.g. the Standard Library, device drivers, middleware, third-party "
            "libraries, automatically generated code, or legacy code) which may or may "
            "not have been developed to comply with The Guidelines. Native code is code "
            "developed within the scope of the current project, developed so as to "
            "comply with The Guidelines applied to the project."
        ),
        "category": "definition",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [25, 39],
        "identifier_heavy": False,
    },
    {
        "question": "Is Standard Library code required to comply with MISRA Guidelines?",
        "ground_truth": (
            "No. As it is part of the compiler's implementation and its functionality is "
            "defined in The Standard, Standard Library code is not required to comply "
            "with MISRA Guidelines. However, guidelines that rely on the interface "
            "provided by standard header declarations and macros are still applicable."
        ),
        "category": "definition",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [27],
        "identifier_heavy": False,
    },
    {
        "question": (
            "What are the four levels of compliance that can be claimed in a guideline "
            "compliance summary?"
        ),
        "ground_truth": (
            "The four levels of compliance that may be claimed for a guideline are: "
            "Compliant (no violations), Deviations (violations supported by "
            "deviations), Violations (violations not supported by deviations), and "
            "Disapplied (no checks were made for compliance)."
        ),
        "category": "classification",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [29],
        "identifier_heavy": True,
    },
    {
        "question": (
            "What artefacts must a supplier deliver to the acquirer to support a claim "
            "of MISRA compliance?"
        ),
        "ground_truth": (
            "On completion of a project, the supplier shall make available: the "
            "guideline enforcement plan (and supporting documentation if requested); "
            "the guideline compliance summary; details of all approved deviation "
            "permits used; and deviation records covering all violations of guidelines "
            "re-categorized as Required."
        ),
        "category": "process",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [30, 31],
        "identifier_heavy": False,
    },
    {
        "question": "What is a guideline re-categorization plan (GRP)?",
        "ground_truth": (
            "A guideline re-categorization plan is a policy agreed between the acquirer "
            "and the supplier at the outset of a project, whereby the MISRA category "
            "assigned to each guideline is reviewed and in some cases superseded by a "
            "more stringent category, determining how The Guidelines are to be applied "
            "to the project."
        ),
        "category": "definition",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [23, 38],
        "identifier_heavy": True,
    },
    {
        "question": "What are the sources adopted code is typically derived from?",
        "ground_truth": (
            "Adopted code is typically derived from: the Standard Library, device "
            "driver files, middleware, third-party libraries, automatically generated "
            "code, and legacy code."
        ),
        "category": "classification",
        "gold_doc": "MISRA-Compliance-2020.pdf",
        "gold_pages": [25],
        "identifier_heavy": False,
    },
    # ── New EpsonPrinterDocumentation.pdf questions ──
    {
        "question": "What should you do if the ink level is below the lower line on an ink tank?",
        "ground_truth": (
            "You need to refill the ink tank, filling it to the upper line. Continued "
            "use of the product when the ink level is below the lower line could damage "
            "the product."
        ),
        "category": "process",
        "gold_doc": "EpsonPrinterDocumentation.pdf",
        "gold_pages": [108, 109],
        "identifier_heavy": False,
    },
    {
        "question": "How do you check the printer's nozzles using the product's buttons?",
        "ground_truth": (
            "Turn the product off, load a few sheets of plain paper, then hold down the "
            "stop button and press the power button to turn the product on. Release "
            "both buttons when the product turns on, and it will print a nozzle check "
            "pattern. Check the printed pattern for gaps in the lines — if there are "
            "gaps or the pattern is faint, the print head needs cleaning."
        ),
        "category": "process",
        "gold_doc": "EpsonPrinterDocumentation.pdf",
        "gold_pages": [119, 120],
        "identifier_heavy": False,
    },
    {
        "question": (
            "What does it mean if the printer's power light is flashing and the ink "
            "light is on?"
        ),
        "ground_truth": (
            "It means an error occurred during firmware updating. You should connect "
            "the product using a USB cable and try updating the firmware again; if the "
            "error continues, contact Epson for support."
        ),
        "category": "definition",
        "gold_doc": "EpsonPrinterDocumentation.pdf",
        "gold_pages": [139],
        "identifier_heavy": True,
    },
    {
        "question": "What should you do if the paper light on the printer is flashing?",
        "ground_truth": (
            "A flashing paper light means paper is jammed in the product. Remove the "
            "jammed paper and press the B&W copy button or the color copy button to "
            "clear the error."
        ),
        "category": "process",
        "gold_doc": "EpsonPrinterDocumentation.pdf",
        "gold_pages": [138],
        "identifier_heavy": True,
    },
    {
        "question": "What should you do if multiple pages feed into the printer at once?",
        "ground_truth": "Remove the paper, fan the edges to separate the sheets, and reload it.",
        "category": "process",
        "gold_doc": "EpsonPrinterDocumentation.pdf",
        "gold_pages": [143],
        "identifier_heavy": False,
    },
    {
        "question": "What is the scanning resolution of the L220's scanner?",
        "ground_truth": "600 dpi for the main scan and 1200 dpi for the sub scan.",
        "category": "definition",
        "gold_doc": "EpsonPrinterDocumentation.pdf",
        "gold_pages": [169],
        "identifier_heavy": True,
    },
    {
        "question": "What interface does the printer use to connect to a computer?",
        "ground_truth": "Hi-Speed USB (Device Class for computers).",
        "category": "definition",
        "gold_doc": "EpsonPrinterDocumentation.pdf",
        "gold_pages": [172],
        "identifier_heavy": True,
    },
    {
        "question": "What is the operating temperature range for the printer?",
        "ground_truth": "50 to 95 °F (10 to 35 °C).",
        "category": "definition",
        "gold_doc": "EpsonPrinterDocumentation.pdf",
        "gold_pages": [172],
        "identifier_heavy": True,
    },
]
