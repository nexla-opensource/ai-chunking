# Extracted verbatim from chunking_code.py — see PROVENANCE.md
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ai_chunking.chunkers.document_chunker")


# ==== Prompts ====
class PromptFactory:
    @staticmethod
    def docmeta(document_text: str, file_title: str, file_type: str = "pdf", custom_instructions: Optional[str] = None) -> str:
        preambles = {
            "pdf": "The input is a document with structured and unstructured content, possibly containing details about various topics, summaries, and data. It may include a contents/index page at the start. Use that to infer the high-level structure and context of the document.",
            "csv": "The input is a CSV data file containing structured tabular data presented in rows and columns with headers describing the data fields. Analyze the data structure, themes, and content patterns within this single data table.",
            "excel": "The input is an Excel spreadsheet file containing one or more sheets of structured data. Each sheet is marked with ######[Sheet: SheetName]###### and contains tabular data with headers. Analyze the data structure, themes, and relationships across all sheets."
        }

        preamble = preambles.get(file_type, preambles["pdf"])

        custom_section = ""
        if custom_instructions:
            custom_section = (
                "\nAdditional guidance for this document set:\n"
                "IMPORTANT: You are running the DOCUMENT-METADATA (docmeta) stage. "
                "Follow ONLY guidance about document-level fields or document_summary "
                "structure. IGNORE any sections titled 'Per-chunk ...' or similar — "
                "those apply to a later chunking stage, not this extraction.\n\n"
                + custom_instructions
                + "\n"
            )

        return f"""
You are an expert information extractor and document summarizer.

{preamble}

This summary will guide further chunking and processing, so ensure accuracy and completeness.

Your task is to extract the following baseline fields from the document:
- "file_title": Use this exact value: "{file_title}".
- "freshness_date": The most relevant date or period mentioned in the document, formatted as YYYY-MM-DD (e.g., if you find "30th June 2023", return "2023-06-30"). If only a month and year are present (e.g., "May 2024"), return the last day of that month (e.g., "2024-05-31"). If only a year is present, return the last day of that year (e.g., "2024-12-31"). Only extract this if it's clearly stated—do not make assumptions. If no date is found, return an empty string.
{custom_section}
STRICT OUTPUT RULES:
- Return a **single valid JSON object**.
- Include the two baseline keys above (file_title, freshness_date).
- Include ONLY additional keys that the guidance above EXPLICITLY names as document-level docmeta fields (e.g. quoted like `"tenant_name": ...`). Do NOT invent keys from table rows, header labels, bullet sub-items, or column names you see in the document.
- Every extra key must be a lowercase snake_case identifier (e.g. `tenant_name`, not `Tenant Name` or `Annual`).
- Output **only** the JSON — no markdown fences, no explanation, no prose before or after.
- Keep the response short; do NOT generate long-form summaries or prose.

Document:
{document_text}
""".strip()

    @staticmethod
    def document_plan(sample_text: str) -> str:
        return f"""
You are a document structure analyst. Examine the sample below (the first few pages of a document) and return a structured plan describing what the full document looks like. Do not chunk, summarize, or extract content — only classify structure.

Return a **single valid JSON object** with exactly these keys:

- "doc_type": free-form label describing the document's nature. Examples: "lease", "financial_report", "technical_spec", "narrative", "legal_contract", "policy_manual", "other".
- "has_headings": true if the document appears to use a heading hierarchy (numbered sections, titled sections, etc.), false if it is flat prose or unstructured.
- "heading_depth_estimate": integer 0-5. Rough depth of the heading hierarchy; 0 when has_headings is false.
- "table_density": one of "none", "low", "medium", "high". How prevalent are tables in the sample?
- "figure_count_estimate": integer. Rough count of charts, figures, or diagrams visible in the sample.
- "scanned_pages_suspected": true if the sample text looks sparse, garbled, or otherwise suggests the PDF is scanned/image-based and OCR is needed.
- "strategy": one of "heading_driven", "density", "hybrid". Your recommendation for chunking strategy based on structure.
- "special_handling": array of strings from this closed set: ["vision_ocr_needed", "tabular_extraction_heavy", "figure_extraction", "legal_structure", "numeric_tables"]. Empty array if none apply.
- "notes": short free-text observation (1-2 sentences) about anything unusual worth knowing for chunking.

Output **only** the JSON, no markdown, no explanation.

Sample:
{sample_text}
""".strip()

    @staticmethod
    def chunking(document_text: str, document_summary: Optional[str], file_type: str = "pdf", custom_instructions: Optional[str] = None) -> str:
        logger.debug(f"Creating chunking prompt for {file_type} - text length: {len(document_text)} chars")
        logger.debug(f"Document summary provided: {bool(document_summary)}")
        if document_summary:
            logger.debug(f"Document summary length: {len(document_summary)} chars")
        if custom_instructions:
            logger.debug(f"Custom instructions provided: {len(custom_instructions)} chars")
        context_descriptions = {
            "pdf": "document text and break it into multiple, highly-structured, context-rich text chunks",
            "csv": "CSV tabular data and break it into multiple, highly-structured, context-rich data chunks that capture the relationships and insights within the structured data",
            "excel": "Excel spreadsheet data with multiple sheets and break it into multiple, highly-structured, context-rich data chunks that capture the relationships and insights within each sheet's structured data"
        }

        context_desc = context_descriptions.get(file_type, context_descriptions["pdf"])

        # Pre-log the prompt creation
        logger.debug(f"Building chunking prompt with context: {context_desc[:50]}...")

        custom_section = ""
        if custom_instructions:
            custom_section = (
                "\nAdditional guidance for this document set:\n"
                "IMPORTANT: You are running the CHUNKING stage (per-chunk output). "
                "Follow ONLY guidance about per-chunk classification fields, tags, or "
                "chunking conventions. IGNORE any sections about document-level fields, "
                "docmeta JSON keys, or document_summary structure — those already ran "
                "in a prior extraction stage, their output is provided as context only.\n\n"
                + custom_instructions
                + "\n"
            )

        prompt = f"""
You are an expert document analyst and assistant. Your task is to process the provided {context_desc} for use in retrieval-augmented generation (RAG) and downstream LLM tasks.

Your output must be a single valid JSON array (list) of objects. Each object must have the following keys (all lower case):

chunk_number: <chunk_number>                # (required, starts from 1 for each page/sheet)
page_number: <page_number>                  # (required for PDF, page number for this chunk)
sheet_name: <sheet_name>                    # (required for Excel, sheet name for this chunk)
universal: <universal_headings>             # (optional, pipe-separated, e.g. "organization name | date | confidential")
section: <parent_section_title>             # (optional, if present)
topic: <main_topic_title>                   # (required, always present)
sub-topics: <comma-separated sub-topics>    # (optional, if present)
section_reference: <exact legal citation>   # (optional, e.g. "Section 3(c)(ii)", "Article 12.4", "Exhibit G")
customer_specific_tags: <tag1>              # (optional, each tag on its own line, if present)
customer_specific_tags: <tag2>
...
chunk_text:
<main content text>                         # (required, from the relevant lines of the document)
summary: <concise summary of the chunk>     # (required)
potential_questions: <json array of 3-5 questions>   # (required, see guidelines below)

---

#### Guidelines for Each Field:

chunk_number:
- Sequential number for chunks within each page/sheet, starting from 1 for each new page/sheet.
- For example: Page 1 chunks would be numbered 1, 2, 3. Page 2 chunks would start over at 1, 2, 3, etc.
- For Excel: Sheet 1 chunks would be numbered 1, 2, 3. Sheet 2 chunks would start over at 1, 2, 3, etc.
- Do not use running totals across pages/sheets.

page_number:
- **Use only for PDF files**. The page number for this chunk.
- **You must set the 'page_number' field to match the [Page X] marker in the input text from which the chunk was created.**
- Do not guess or skip page numbers. If a chunk is created from text under [Page 4], its page_number must be "4" (as a string).
- **Omit this field for CSV and Excel files**.

sheet_name:
- **Use only for Excel files**. The sheet name for this chunk.
- **You must set the 'sheet_name' field to match the [Sheet: SheetName] marker in the input text from which the chunk was created.**
- Do not guess or skip sheet names. If a chunk is created from text under [Sheet: Holdings], its sheet_name must be "Holdings" (as a string).
- **Omit this field for PDF and CSV files**.

universal:
- List all universal headings relevant to the document or chunk, separated by `|`.
- These may include: fund name, document type, date, confidentiality status, or any other heading that applies to the entire document.
- If no universal headings are present, omit this line.

section:
- The parent section title, if the chunk is a sub-section or belongs to a larger section.
- Omit if not applicable.

topic:
- The main topic or title of this chunk (usually the section or sub-section title).
- Always present.

sub-topics:
- If this chunk merges multiple sub-sections or covers several sub-topics, list them here, comma-separated.
- Omit if not applicable.

section_reference:
- The exact legal citation or numbering for this chunk's content (e.g. "Section 3(c)(ii)", "Article 12.4", "Schedule F, Section 2", "Exhibit G", "Rule 14").
- This is distinct from 'section' (which is a human-readable heading). section_reference is the precise numbering the user would cite.
- Omit if no legal numbering is present.

customer_specific_tags: <json array of strings>   # (optional, e.g. ["As of 2024-04-01","Internal"])
- Omit if not applicable.
- If the "Additional guidance for this document set" section (below) defines *per-chunk* classification fields or tag conventions, follow those and emit the corresponding fields on each chunk.
- DO NOT emit document-level fields (tenant names, doc type, effective date, etc.) inside customer_specific_tags or any other chunk field. Those belong to the docmeta stage, not chunk output.

potential_questions: <json array of 3-5 question strings>   # (required)
- Imagine a user querying a RAG system and generate 3-5 questions this chunk directly and completely answers.
- Each question must be answerable ONLY from the text of this chunk, not from general knowledge or neighboring chunks.
- Write questions as a user would phrase them (natural language, not keywords).
- Cover different aspects of the chunk — if the chunk has a date, amount, and condition, write one question for each rather than five variations on the same fact.
- Example: for a chunk about a security deposit amount and burn-down schedule, good questions include "What is the security deposit amount?", "Under what conditions does the security deposit reduce over time?", and "When does the burn-down begin?". A bad question would be "What does this lease say?" (too broad; not chunk-specific).

chunk_text:
- Be between 70 to 330 tokens.
- Create a well-structured, readable version of the chunk content with proper connective words and flow.
- Integrate headers and subheaders naturally into the text flow, using clear transitions.
- For tables, convert to readable prose that describes the data structure, key values, and relationships. Include all important data points but make them accessible in paragraph form.
- For footnotes, seamlessly integrate the footnote information into the main text where it adds value, using phrases like "It should be noted that..." or "Additionally..." rather than keeping raw markers.
- Transform lists into flowing paragraphs with appropriate connective phrases like "Furthermore," "Additionally," "Moreover," etc.
- For images, charts, or visuals, incorporate any available captions or descriptions naturally into the text.
- Ensure all factual information is preserved while improving readability and coherence.
- Use clear, professional language that maintains the original meaning while enhancing comprehension.
- CRITICAL COVERAGE RULE: Every paragraph, subsection, table, and list in the input MUST appear in at least one chunk. If a page's content exceeds 330 tokens, create multiple chunks to cover it all. Never skip or drop content to stay within the token limit — instead, create additional chunks.
- When a page has both narrative text AND structured content (tables, lists, schedules), create separate chunks for each rather than merging them into one chunk that drops content.

summary:
- Provide a concise, high-level summary that captures the key points and main message of the content field.
- Base the summary exclusively on the information presented in the content field above - do not add external knowledge or interpretation.
- Focus on the most important insights, conclusions, or data points mentioned in the content.
- Keep it brief but comprehensive enough to give a reader a clear understanding of what this chunk covers.
- Use clear, accessible language that would help in retrieval and search scenarios.

#### Chunking Density Rule:
- For dense pages with multiple subsections, tables mixed with paragraphs, or legal clauses with sub-clauses, you MUST produce multiple chunks. There is no maximum chunk count per page.
- A single dense page may require 3, 5, or even 10+ chunks. This is expected and correct.
- Each chunk should cover one coherent logical unit: a single subsection, a single table, a single clause, or a group of closely related short items.
- NEVER sacrifice content completeness to minimize chunk count. More chunks with complete coverage is always better than fewer chunks with dropped content.
- After generating all chunks for a block, verify: does every paragraph and every table from the input appear in at least one chunk? If not, add the missing content as additional chunks.

General Instructions:
- Be explicit and descriptive: Use clear, unambiguous language for all fields.
- Transform content for readability: Content should be well-structured prose, not raw extracted text. Headings and metadata should be integrated naturally into flowing text.
- Complete content coverage: Every piece of information in the source text must appear in at least one chunk. Do not drop, skip, or omit any paragraphs, subsections, table rows, or data points. If content does not fit in existing chunks, create additional chunks.
- No extraneous commentary: Do not add explanations, comments, or formatting outside the specified structure.
- If a field is not applicable, omit it: Do not include empty or placeholder lines.
- All field names must be lower case.
- Only generate the output as a list of objects, nothing else, dont have any other text outside the list of objects.

---

Checklist Before Submitting:
- All required fields are present and correctly labeled.
- CONTENT COMPLETENESS: Every paragraph, subsection, table, and list item from the input is represented in at least one chunk. No content has been dropped or skipped.
- If a page has both narrative text and tables/lists, verify BOTH are covered in separate chunks.
- No extraneous or missing information.
- The format matches the examples exactly.

---
Example Output:
Output is expected to be a list of objects

**SAMPLE OUTPUT FOR PDF:**
[{{
    "chunk_number": "1",
    "page_number": "1",
    "universal": "organization name | date | confidential",
    "section": "Parent Section Title",
    "topic": "Subsection Topic",
    "sub_topics": "sub-topic 1, sub-topic 2",
    "section_reference": "Section 3(a)",
    "customer_specific_tags": ["tag 1","tag 2"],
    "chunk_text": "Well-structured, flowing prose covering the first subsection's narrative content. Integrates headers and context naturally with connective words and clear transitions. All factual information from this subsection is preserved.",
    "summary": "Concise summary of the subsection's content.",
    "potential_questions": ["What specific question is answered here?", "What value or condition is defined?", "When does this apply?"]
}},
{{
    "chunk_number": "2",
    "page_number": "1",
    "universal": "organization name | date | confidential",
    "section": "Parent Section Title",
    "topic": "Second Subsection Topic",
    "sub_topics": "sub-topic A, sub-topic B",
    "section_reference": "Section 3(b), Exhibit B",
    "customer_specific_tags": ["tag 1","tag 2"],
    "chunk_text": "Detailed prose representation of a table, converting each row into readable text with all values preserved. All data points from the table are included in flowing paragraph form.",
    "summary": "Summary of the tabular content including the relationships between values.",
    "potential_questions": ["What are the table's key values?", "How do row A and row B relate?", "What is the total/summary across rows?"]
}}]

**SAMPLE OUTPUT FOR EXCEL:**
[{{
    "chunk_number": "1",
    "sheet_name": "Holdings Q1",
    "universal": "organization name | date | confidential",
    "section": "section title",
    "topic": "topic title",
    "sub_topics": "sub topic 1, sub topic 2",
    "customer_specific_tags": ["tag 1","tag 2"],
    "chunk_text": "Well-structured, flowing prose that describes the data structure, key values, and relationships from the spreadsheet. All important data points are made accessible in paragraph form.",
    "summary": "Brief summary capturing the essence of this specific data chunk."
}}]

**SAMPLE OUTPUT FOR CSV:**
[{{
    "chunk_number": "1",
    "universal": "organization name | date | confidential",
    "section": "section title",
    "topic": "topic title",
    "sub_topics": "sub topic 1, sub topic 2",
    "customer_specific_tags": ["tag 1","tag 2"],
    "chunk_text": "Well-structured, flowing prose that describes the tabular data structure, key values, and relationships. All important data points are made accessible in paragraph form.",
    "summary": "Brief summary capturing the essence of this specific data chunk."
}}]

**END OF SAMPLE OUTPUT**

Document summary for context:
{document_summary if document_summary else ''}
{custom_section}
Document text to process:
{document_text}
"""

        # Log final prompt details
        final_prompt_length = len(prompt)
        logger.debug(f"Final chunking prompt created - length: {final_prompt_length} chars")
        if final_prompt_length > 100000:  # Warn if prompt is very large
            logger.warning(f"Very large prompt created ({final_prompt_length} chars) - may cause issues")

        return prompt.strip()

    @staticmethod
    def vision_enrich_prompt(extracted_text: str) -> str:
        return f"""
Analyze this PDF page image to identify and describe visual elements that supplement the extracted text.

EXTRACTED TEXT FROM PAGE:
{extracted_text}

ANALYSIS TASK:
Examine the image for:
1. Charts, graphs, tables, diagrams, or data visualizations
2. Formatting, layout, colors, or visual structure that conveys meaning
3. Images, logos, icons, or visual branding elements
4. Any text or numbers visible in the image that weren't captured in extraction
5. Spatial relationships, positioning, or visual hierarchy of elements

INSTRUCTIONS:
- Describe ONLY what you can see in the image
- Include specific details: numbers, colors, positioning, sizes, trends
- Focus on visual elements that add context or meaning beyond the extracted text
- Be comprehensive but factual - don't infer or speculate
- If charts/tables are present, describe their structure and key data points
- Note any formatting or layout that affects document meaning

RESPONSE FORMAT:
<enriched_text>
[Provide detailed descriptions of visual elements. Include specific values, colors, positioning, and relationships you observe. If no significant visual elements are present, state "No significant visual elements detected - page appears to be primarily text-based."]
</enriched_text>

<classification>
[NECESSARY if the image contains charts, graphs, tables, or visual elements that add important information]
[UNNECESSARY if the page is primarily text without significant visual elements]
</classification>

Provide complete descriptions in both sections.
"""

    @staticmethod
    def vision_batch_prompt(num_pages: int) -> str:
        """Prompt for a batched vision call: N page images -> JSON array of N results."""
        return f"""
You are processing {num_pages} scanned/visual PDF pages, attached below in order. For each image, perform full OCR and content extraction.

INSTRUCTIONS (apply to each page):
1. Transcribe every piece of text visible on the page, preserving reading order (top-to-bottom, left-to-right).
2. For tables: reproduce the structure as plain text (pipe- or tab-separated columns).
3. For charts/graphs: describe the chart type, axis labels, data values, and key takeaways.
4. For headers/footers/page numbers: include them, marked as "[Header]" / "[Footer]".
5. For handwritten text: transcribe as best you can, noting "[handwritten]".
6. Preserve paragraph breaks and section headings.
7. Do NOT summarize or paraphrase — transcribe faithfully.

OUTPUT FORMAT — return a JSON array with EXACTLY {num_pages} objects, in the same order as the input images:

[
  {{"page_index": 1, "enriched_text": "<full transcription of image 1>", "classification": "NECESSARY"}},
  {{"page_index": 2, "enriched_text": "<full transcription of image 2>", "classification": "NECESSARY"}},
  ...
]

Rules:
- page_index is 1-based and matches image order.
- classification: "NECESSARY" when the page has substantive content; "UNNECESSARY" only for blank or purely decorative pages.
- Return ONLY the JSON array. No markdown fences, no prose before or after.
""".strip()

    @staticmethod
    def vision_ocr_prompt() -> str:
        return """
This PDF page has NO machine-readable text layer — it is a scanned/image-based page.
Your task is to perform FULL OCR: extract ALL text visible on this page image.

INSTRUCTIONS:
1. Transcribe every piece of text you can see, preserving the reading order (top-to-bottom, left-to-right).
2. For tables: reproduce the table structure as clearly as possible using plain text (pipe-separated or tab-separated columns).
3. For charts/graphs: describe the chart type, axis labels, data values, and any trends or key takeaways.
4. For headers/footers/page numbers: include them but mark them clearly (e.g., "[Header]", "[Footer]").
5. For handwritten text: transcribe as best you can; note "[handwritten]" if legibility is uncertain.
6. Preserve paragraph breaks and section headings.
7. Do NOT summarize or paraphrase — transcribe the actual text as-is.

RESPONSE FORMAT:
<enriched_text>
[Complete transcription of all text visible on the page. Include table data, chart descriptions, and all readable content.]
</enriched_text>

<classification>
NECESSARY
</classification>

Provide the complete transcription in the enriched_text section.
"""

    @staticmethod
    def header_extraction(table_text: str, file_type: str = "table") -> str:
        """Generate prompt for extracting clean headers from tabular data"""
        return f"""
Extract the column headers from this {file_type} data and return them as a JSON array.

RULES:
1. Look for the header row(s) - usually the first row but may be later
2. If there are multiple header rows, combine them with "/" (e.g., "Quarter/2023_Q1")
3. Clean up formatting, remove extra spaces and special characters
4. Return exactly what the columns should be called
5. MUST return valid JSON array format

DATA:
```
{table_text}
```

RESPONSE FORMAT:
Return ONLY a JSON array like this example:
["Investment Name", "Asset Type", "Date", "Amount", "Status"]

Do not include any other text, explanations, or formatting. Only the JSON array.
""".strip()

    @staticmethod
    def filename_date_extraction(file_name: str) -> str:
        """Generate prompt for extracting dates from filenames"""
        return f"""
Extract the date from this filename: "{file_name}"

Examples:
- "Report_2024-08-31.xlsx" → 2024-08-31
- "Sales_June_2023.csv" → 2023-06-30
- "Q1_2024_data.xlsx" → 2024-03-31
- "transactions_20240515.csv" → 2024-05-15
- "annual_report_2023.pdf" → 2023-12-31

Answer with just the date in YYYY-MM-DD format, or "Unknown" if no date found.
""".strip()

    @staticmethod
    def tabular_chunking(table_text: str, start_row: int, end_row: int, file_type: str = "CSV", sheet_name: str = None, custom_instructions: Optional[str] = None) -> str:
        """Generate detailed prompt for tabular chunk processing with comprehensive guidelines"""

        # Determine context based on file type
        if file_type.upper() == "CSV":
            context_desc = "CSV tabular data and break it into multiple, highly-structured, context-rich data chunks that capture the relationships and insights within the structured data"
            location_field = ""
            location_guidance = ""
        else:  # Excel
            context_desc = "Excel spreadsheet data and break it into multiple, highly-structured, context-rich data chunks that capture the relationships and insights within the structured data"
            location_field = "sheet_name: <sheet_name>                    # (required for Excel, sheet name for this chunk)"
            location_guidance = f"""
sheet_name:
- **Use only for Excel files**. The sheet name for this chunk.
- For this chunk, use: "{sheet_name}"
- **Omit this field for CSV files**.
"""

        custom_section = ""
        if custom_instructions:
            custom_section = (
                "\nAdditional guidance for this document set:\n"
                + custom_instructions
                + "\n"
            )

        return f"""
You are an expert data analyst and assistant. Your task is to process the provided {context_desc} for use in retrieval-augmented generation (RAG) and downstream LLM tasks.

This data block contains rows {start_row} to {end_row} of the dataset.

Your output must be either:
- A single JSON object (for simple data blocks), OR
- A JSON array of objects (for complex data blocks that need multiple chunks)

Each chunk object must have the following keys (all lower case):

chunk_number: <chunk_number>                # (required, block sequence number)
{location_field}
summary: <concise summary of the data>      # (required, top-level key for searchability)
topic: <main_topic_title>                   # (required, descriptive title for this data block)
chunk_text: <detailed analytical prose>     # (required, comprehensive data analysis with connective words)
content_type: "metrics"                     # (required, always "metrics" for tabular data blocks unless clearly otherwise)
potential_questions: <json array>           # (required, 3-5 specific questions this chunk's data directly answers; see guidelines)
universal: <universal_headings>             # (optional, pipe-separated, e.g. "dataset name | date range | data source")
section: <parent_section_title>             # (optional, if this block belongs to a larger data section)
section_reference: <citation if present>    # (optional, e.g. "Schedule A", "Table 3.2" — exact reference string from the data, or "")
sub_topics: <comma-separated sub-topics>    # (optional, if this block covers multiple data categories)
customer_specific_tags: <tag1>              # (optional, each tag on its own line, if present)
customer_specific_tags: <tag2>
...

Additional per-chunk fields:
- If the "Additional guidance for this document set" section below declares extra per-chunk fields (e.g. via `"field_name":`), include them as additional keys. Use lowercase snake_case.

---

potential_questions guidelines:
- Write 3-5 natural-language questions a user might ask to retrieve THIS specific data block.
- Each question must be answerable ONLY from this block's data — not general knowledge, not adjacent blocks.
- Be specific to the values in this block: "What was total revenue in Q3 2024?" not "What does the data say?"
- For a block with dates, currencies, or named entities, prefer questions referencing those values.

----
IMPORTANT:
- For simple/sparse data blocks: Return a single JSON object
- For complex/dense data blocks: Return a JSON array of multiple chunk objects
- Create multiple chunks when you have diverse topics, or when a single chunk_text would exceed 400 tokens
- Each chunk should focus on a specific aspect, pattern, or subset of the data

#### Guidelines for Each Field:

chunk_number:
- Sequential number for this data block within the larger dataset.
- Start from 1 and increment for each block.

{location_guidance}
summary:
- Provide a concise, high-level summary that captures the key characteristics, patterns, and insights from the data block.
- This is a top-level field for searchability and discovery.
- Focus on the most important data patterns, distributions, trends, or notable characteristics.
- Include data volume, completeness, and quality indicators where relevant.
- Keep it brief but comprehensive enough to give a reader a clear understanding of what this data block contains.
- Use clear, accessible language optimized for search and retrieval scenarios.

topic:
- The main topic or descriptive title of this data block (e.g., "Customer Demographics Data", "Financial Transactions Q1", "Inventory Levels by Region").
- Always present and should clearly indicate what kind of data this block contains.

chunk_text:
- Be between 70 to 400 tokens.
- Extract the exact values from the table without summarizing, generalizing, or inferring patterns.
- Convert all rows and columns into flowing prose where each value is explicitly included.
- Use connective words and phrases like "The data shows that...", "Additionally...", "Furthermore...", "Moreover..." to link the values together smoothly.
- Integrate column headers directly with their corresponding values in natural sentences, ensuring no information is dropped.
- Preserve the original detail and order of the table so the text contains the same information as the CSV.
- Do not describe distributions, ranges, or relationships—only state the values as they appear.
- For missing or null values, include them explicitly with phrasing like: "This entry has no value for [column name]..."
- Maintain professional and clear language while ensuring the prose is a complete textual representation of the table.

universal:
- List all universal headings relevant to the dataset or data block, separated by `|`.
- These may include: dataset name, data source, collection date, data type, or any other metadata that applies to the entire dataset.
- If no universal headings are present, omit this line.

section:
- The parent section title, if this data block belongs to a larger categorical section within the dataset.
- Omit if not applicable.

sub_topics:
- If this data block covers multiple sub-categories or data types, list them here, comma-separated.
- For example: "Age Groups, Income Brackets, Geographic Distribution"
- Omit if not applicable.

customer_specific_tags: <json array of strings>   # (optional, e.g. ["2024-Q1","Internal Use","Validated"])
- Omit if not applicable.

General Instructions:
- Be explicit and descriptive: Use clear, unambiguous language for all fields that accurately represents the data.
- Transform data for analytical readability: Content should be well-structured analytical prose, not raw table descriptions. Headers and data relationships should be integrated naturally into flowing text.
- Comprehensive data coverage: Ensure all relevant data patterns, relationships, and insights are captured and presented in the content field with proper analytical context.
- Preserve data integrity: Do not alter, interpret, or add to the actual data values - only transform the presentation format.
- No extraneous commentary: Do not add explanations, comments, or formatting outside the specified structure.
- If a field is not applicable, omit it: Do not include empty or placeholder lines.
- All field names must be lower case.
- Only generate the output as a single JSON object, nothing else. Do not include any other text outside the JSON object.

---

Data Block:
```
{table_text}
```
{custom_section}
EXAMPLES:
Simple data (single object): {{"chunk_number": "1", "summary": "...", "chunk_text": "...", "topic": "..."}}
Complex data (array): [{{"chunk_number": "1", "summary": "...", "chunk_text": "...", "topic": "..."}}, {{"chunk_number": "2", "summary": "...", "chunk_text": "...", "topic": "..."}}]

Return only the JSON object or array, no additional text.
""".strip()

    @staticmethod
    def image_analysis_prompt() -> str:
        """Generate prompt for general image analysis."""
        return """
You are an expert image analyst. Your task is to perform a comprehensive analysis of the provided image and return the results in JSON format.

INSTRUCTIONS:
1. **Describe the Scene:** What is the overall setting or context of the image (e.g., office, nature, screenshot of a software application)?
2. **Identify Objects:** List and describe all significant objects, people, or animals visible.
3. **Extract Text:** Transcribe ALL text visible in the image, preserving its original formatting as best as possible. If there is no text, state "No text is visible."
4. **Analyze Data/Charts:** If the image contains a chart, graph, or table, describe its type (e.g., bar chart, pie chart), explain what data it represents, and extract key data points, trends, or conclusions.
5. **Summarize the Purpose:** Based on all the elements, provide a concise summary of the image's likely purpose or the main message it conveys.

Return your response as a JSON object with exactly these two keys:
{
  "summary": "A concise 1-2 sentence summary of the image's main content and purpose",
  "chunk_text": "A comprehensive, detailed analysis covering all the above points in a well-structured narrative"
}

Return only the JSON object, no additional text.
""".strip()

    # ==== Phase 4: heading-driven chunking prompts ====

    @staticmethod
    def heading_detection(numbered_text: str) -> str:
        return f"""
You are a document-structure analyst. The text below has each line prefixed with its absolute line number in the format:

    <line_num> | <line content>

Your ONLY task: identify lines that act as HEADINGS and classify each one.

HEADING TYPES (required per heading):

- "global"  — document-wide context that applies to every chunk: fund names, organization names, report period, document title, overall confidentiality marks. Usually appears at the document start, repeated across pages, or in running headers.
- "page"    — a page-level topic that organizes one page's content (e.g., "Risk Metrics Analysis" at the top of a new page). Falls under global context.
- "section" — hierarchical structure within the body: articles, numbered sections, named clauses, sub-sections (1.1, (a), SCHEDULE A, etc.). These are the main chunk boundaries.
- "footer"  — repeated metadata, disclaimers, page numbers, copyright/attribution lines. Not real content boundaries; should be kept separate from chunks.
- "bridge"  — a heading that EXPLICITLY continues a topic from the previous page/block. Use SPARINGLY, only when at least one of these is true:
    * the heading has an explicit continuation marker: "(continued)", "(cont.)", "— cont.", "Continuation of ...", "(cont'd)"
    * the heading is a sub-heading whose enumerator is mid-sequence (e.g. "(d)" with no "(a)/(b)/(c)" visible on this page, making it clear the parent enumerator started earlier)
  If a heading introduces a fresh topic (even if it appears right after a page break), classify it as "section", NOT "bridge". Default to "section" when uncertain — over-labeling bridges hurts downstream chunking.

GENERAL RULES:
- Do NOT include body text, paragraphs, tables, signatures, or inline footnotes as headings.
- Do NOT rewrite or reword the heading text — quote it verbatim from the source.
- Do NOT emit any content beyond what's asked. No chunks. No summaries.
- A page marker like "######[Page N]######" is NOT a heading.
- Give each heading a numeric depth level (for section/page types):
  - level=1: top-level (article, major numbered section, chapter)
  - level=2: sub-section (e.g., "3.1", "(a)", named clauses within a section)
  - level=3: sub-sub-section (rare; only if clearly hierarchical)
  For global/footer/bridge types, level is still required but usually 1.
- If the document appears to have no structural headings at all (flat narrative), return an empty array.

CONTENT-TYPE HINT (per heading, required for section/page/bridge; optional for global/footer):

- "content_type": EXACTLY one of: "metrics" (quantitative data/tables/numbers), "narrative" (explanatory prose), "instructions" (step-by-step procedure), "comparisons" (parallel structures comparing items), "legal" (disclaimers, clauses, formal legal language), "concept" (term definitions), "other".
  This is YOUR best guess at what content follows this heading. It need not be perfect — it's used downstream to decide whether the section must be kept as its own chunk.
  For global/footer headings, you may set "other".

OUTPUT: a single JSON array of objects, ordered by ascending line_num:

[
  {{"line_num": 2,  "heading_text": "Oaktree Global Credit Fund — 2024 Q2 Report", "level": 1, "heading_type": "global",  "content_type": "other"}},
  {{"line_num": 12, "heading_text": "ARTICLE I — PREMISES",                         "level": 1, "heading_type": "section", "content_type": "legal"}},
  {{"line_num": 23, "heading_text": "1.1 Description of Premises",                   "level": 2, "heading_type": "section", "content_type": "legal"}},
  {{"line_num": 41, "heading_text": "Rent Schedule",                                 "level": 2, "heading_type": "section", "content_type": "metrics"}},
  {{"line_num": 67, "heading_text": "ARTICLE I — PREMISES (continued)",              "level": 1, "heading_type": "bridge",  "content_type": "legal"}},
  {{"line_num": 99, "heading_text": "Page 4 of 20",                                  "level": 1, "heading_type": "footer",  "content_type": "other"}}
]

Return ONLY the JSON array — no markdown fences, no prose.

---
TEXT WITH LINE NUMBERS:
{numbered_text}
""".strip()

    @staticmethod
    def universal_headings_dedupe(global_headings: List[Dict[str, Any]],
                                   footer_headings: List[Dict[str, Any]],
                                   source_info: str = "") -> str:
        g = json.dumps(global_headings[:50], default=str)  # cap to keep prompt bounded
        f = json.dumps(footer_headings[:50], default=str)
        return f"""
You are given two types of headings extracted from a multi-page document, plus the document's source info. Produce a single canonical list of UNIVERSAL headings — headings that apply to the entire document and should be attached to every chunk for retrieval.

**Global Headings** (pre-classified as document-wide):
{g}

**Footer Headings** (may include disclaimers, page labels, repeated structural notes — most are NOT universal):
{f}

**Source info** (document path/filename — may contain clues like fund name, report period):
{source_info!r}

Your task:

1. **Deduplicate semantically.** "LEASE", "Lease Agreement", and "LEASE (continued)" should collapse to one heading. "US", "USA", and "United States of America" are the same. Collapse alternate phrasings, abbreviations, and punctuation variants.

2. **Keep only TRULY universal headings.** A heading is universal if it applies to the entire document's context. Examples:
   - Document title (e.g. "2024 Annual Report")
   - Organization / fund / party name
   - Document-wide date or reporting period

   Reject headings that are:
   - Page-specific topics (e.g. "Page 4 Notes", "Q2 2024 Metrics")
   - Per-page running footers with page numbers ("Page 2 of 20")
   - Disclaimers that apply only to a specific section

3. **You may consult source_info** to disambiguate. If the filename contains a fund name or date that isn't captured in the extracted headings, add it.

Return a JSON object:

{{
  "universal_headings": [
    {{"title": "Canonical Heading 1"}},
    {{"title": "Canonical Heading 2"}}
  ]
}}

Return ONLY the JSON — no markdown, no prose.
""".strip()

    @staticmethod
    def table_extraction_batch(chunks: List[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for i, c in enumerate(chunks):
            blocks.append(
                f"### CHUNK {i + 1} (topic={c.get('topic', '')!r}, "
                f"section={c.get('section', '')!r})\n"
                f"{(c.get('chunk_text') or '')[:4000]}"
            )
        joined = "\n\n".join(blocks)
        return f"""
You are a structured-data extractor. Each chunk below contains one or more tables, numeric schedules, or tabular data. Extract the structured tables from each chunk's text.

Output requirements per chunk:

- Identify every table. A "table" is a group of rows with a consistent column structure — rent schedules, operating expense breakdowns, numerical comparisons, etc. A single chunk may have zero, one, or multiple tables.
- For each table, emit:
  - "title": a short label describing what the table represents (inferred from the chunk's heading, caption, or column content).
  - "headers": array of column names. If the source text uses pipe/tab-separated text, use those. If the text is prose describing a table, infer the columns.
  - "rows": array of arrays, each inner array a list of cell values in the same order as `headers`. All values as strings; preserve units, currency symbols, and formatting verbatim (e.g. "$1,250", "4.5%", "2024-01-01").
  - "notes": optional free-text explaining edge cases (merged cells, missing values, footnote references).
- If a chunk contains NO extractable tables (it's pure prose, or its numbers are embedded in sentences without tabular structure), return `"tables": []` for that chunk.
- Do NOT invent rows or values. If the source text is ambiguous (e.g. missing a cell value), omit that row and note the ambiguity in `notes`.
- Do NOT return the chunk's prose text — only the structured tables.

Return a JSON array with EXACTLY {len(chunks)} entries in input order:

[
  {{
    "chunk_index": 1,
    "tables": [
      {{
        "title": "Base Rent Schedule",
        "headers": ["Year", "Base Rent (Annual)", "Per RSF"],
        "rows": [
          ["1",  "$120,000.00", "$40.00"],
          ["2",  "$123,600.00", "$41.20"]
        ],
        "notes": ""
      }}
    ]
  }},
  {{
    "chunk_index": 2,
    "tables": []
  }}
]

Return ONLY the JSON array — no markdown fences, no prose.

---
{joined}
""".strip()

    @staticmethod
    def chunk_verification_batch(chunks: List[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for i, c in enumerate(chunks):
            pq = c.get("potential_questions") or []
            if isinstance(pq, str):
                pq = [pq]
            blocks.append(
                f"### CHUNK {i + 1}\n"
                f"content_type: {c.get('content_type', 'other')!r}\n"
                f"topic: {c.get('topic', '')!r}\n"
                f"section_reference: {c.get('section_reference', '')!r}\n"
                f"summary: {c.get('summary', '')!r}\n"
                f"potential_questions: {json.dumps(pq, default=str)}\n"
                f"--- chunk_text ---\n"
                f"{(c.get('chunk_text') or '')[:3000]}"
            )
        joined = "\n\n".join(blocks)
        return f"""
You are a strict RAG quality auditor. You are given {len(chunks)} chunks, each with its generated metadata (summary / topic / content_type / section_reference / potential_questions) AND its actual chunk_text.

Your job: for each chunk, verify that the METADATA is GROUNDED in chunk_text. Metadata is grounded when:

- Every fact asserted in `summary` is supported by chunk_text. No external knowledge. No invented dates, parties, numbers.
- Every `potential_questions` item is answerable ONLY from chunk_text (not general knowledge, not neighboring chunks).
- `content_type` matches the dominant nature of chunk_text (metrics = quantitative/tables, narrative = explanatory prose, instructions = procedural, comparisons = parallel structures, legal = disclaimers/clauses, concept = definitions, other = catchall).
- If `section_reference` is non-empty, the referenced section identifier (e.g. "Section 3(a)", "Article II") must actually appear in chunk_text.

Return a JSON array with EXACTLY {len(chunks)} entries, in order:

[
  {{
    "chunk_index": 1,
    "grounded": true,
    "issues": [],
    "severity": "ok"
  }},
  {{
    "chunk_index": 2,
    "grounded": false,
    "issues": ["summary mentions 'Q3 2024 revenue' but no revenue figure appears in chunk_text", "potential_questions[1] asks about a tenant not named in chunk"],
    "severity": "major"
  }}
]

`severity` is one of: "ok" (grounded), "minor" (small drift, usable), "major" (significant hallucination, re-generate).

Return ONLY the JSON array — no markdown, no prose.

---
{joined}
""".strip()

    @staticmethod
    def chunk_metadata_batch_retry(chunks: List[Dict[str, Any]], issues_by_index: Dict[int, List[str]],
                                   doc_summary: Optional[str] = None,
                                   custom_instructions: Optional[str] = None) -> str:
        chunk_blocks: List[str] = []
        for i, c in enumerate(chunks):
            flagged = issues_by_index.get(i, [])
            issues_hint = ""
            if flagged:
                issues_hint = "\n# PRIOR ISSUES TO FIX:\n" + "\n".join(f"#   - {x}" for x in flagged)
            chunk_blocks.append(
                f"### CHUNK {i + 1} (section_heading={c.get('heading_text', '')!r}, "
                f"page_number={c.get('page_number', '')})"
                + issues_hint
                + "\n"
                + (c.get('chunk_text', '') or "")
            )
        joined = "\n\n".join(chunk_blocks)

        doc_summary_section = ""
        if doc_summary:
            doc_summary_section = f"\nDocument-level summary for context:\n{doc_summary}\n"

        custom_section = ""
        if custom_instructions:
            custom_section = (
                "\nAdditional guidance for this document set:\n"
                "IMPORTANT: You are running the METADATA-RETRY stage. The prior metadata generation "
                "failed verification for one or more chunks (see PRIOR ISSUES per chunk). Do NOT repeat "
                "those mistakes. chunk_text is FIXED — do not alter or regenerate it.\n\n"
                + custom_instructions
                + "\n"
            )

        return f"""
You are a strict RAG metadata generator. Below are {len(chunks)} chunks whose earlier metadata was rejected by a verifier. Generate new metadata that is STRICTLY grounded in each chunk's text.

Rules (failure to follow = rejection):

1. Every word of `summary` must be traceable to specific text in the chunk. If a chunk is sparse, write a sparse summary — DO NOT invent facts.
2. Every `potential_questions` item must be answerable ONLY from this chunk's text. Ban questions requiring general knowledge or neighboring chunks.
3. `content_type` must match chunk_text's dominant nature. Do not guess.
4. `section_reference` must be a string that literally appears in chunk_text, or "" if none.
5. If PRIOR ISSUES are listed for a chunk, explicitly address each.

Per-chunk output fields (all required):
- "chunk_index": 1-based, matching the CHUNK N label.
- "topic": concise 3-10 words grounded in chunk_text.
- "summary": 1-2 sentences, each traceable to chunk_text.
- "sub_topics": comma-separated or "".
- "section_reference": literal citation found in chunk_text, or "".
- "content_type": metrics|narrative|instructions|comparisons|legal|concept|other.
- "potential_questions": array of 3-5 specific questions answerable from chunk_text alone.
- "customer_specific_tags": array per guidance below, or [].
- "universal": pipe-separated, or "".

Additional per-chunk fields:
- If the guidance below declares extra per-chunk fields, include them (lowercase snake_case keys).
{doc_summary_section}{custom_section}
Return a JSON array with EXACTLY {len(chunks)} objects, same order. chunk_text is not returned. No markdown, no prose before/after.

---
{joined}
""".strip()

    @staticmethod
    def bridge_classification(current_page_last_heading: str,
                              current_page_last_heading_text: Dict[int, str],
                              next_page_first_heading_text: Dict[int, str],
                              current_page_headings: List[Dict[str, Any]],
                              universal_headings: List[str]) -> str:
        last_text = json.dumps(current_page_last_heading_text, default=str)[:4000]
        prelude = json.dumps(next_page_first_heading_text, default=str)[:4000]
        current_headings_json = json.dumps(
            [{"title": h.get("heading_text"), "line_num": h.get("line_num")}
             for h in current_page_headings[-20:]],
            default=str,
        )
        uh = json.dumps(universal_headings[:20], default=str)

        return f"""
You have information about two consecutive parts of a document:

**Previous Block's Final Heading**
- Title: {current_page_last_heading!r}
- Lines directly under that heading (line_number → text):
{last_text}

**Next Block — Lines BEFORE its first explicit heading** (the "prelude"):
{prelude}

**Universal Headings** (apply document-wide):
{uh}

**Previous Block — All Headings** (for "relevant" candidates):
{current_headings_json}

### Task

Decide if **any** lines in the next block's prelude logically continue the previous block's final heading.

- **bridging = "true"** — at least some prelude lines extend the topic of the previous block's final heading.
- **bridging = "relevant"** — prelude lines don't continue the final heading, but partially reference OTHER headings from the previous block. List which in `most_relevant_headings`.
- **bridging = "false"** — prelude is a fresh topic, or is fully explained by universal headings / disclaimers.

### Steps

1. **Universal check.** If prelude text matches the universal headings (e.g. document title, dates), it doesn't bridge from body-content headings.
2. **Final-heading check.** Does prelude text continue the subject of the previous block's final heading? If yes → `"true"`, and set `start_index` to the first prelude line that picks up the topic.
3. **Relevance check.** If not continuing the final heading, does prelude text reference other headings from the previous block? If yes → `"relevant"` with `most_relevant_headings`.
4. **Otherwise** → `"false"`.

### Edge Cases
- Partial bridging: lines 1–2 disclaimers, line 3 continues → `"true"`, `start_index: 3`, rationale explains.
- Empty prelude → `"false"`.

### Output
Return ONLY this JSON object:

{{
  "bridging": "true" | "false" | "relevant",
  "start_index": <integer or null>,
  "rationale": "<brief explanation>",
  "most_relevant_headings": ["heading 1", ...]
}}
""".strip()

    @staticmethod
    def chunk_metadata_batch(chunks: List[Dict[str, Any]], doc_summary: Optional[str] = None,
                             custom_instructions: Optional[str] = None) -> str:
        chunk_blocks = []
        for i, c in enumerate(chunks):
            chunk_blocks.append(
                f"### CHUNK {i + 1} (section_heading={c.get('heading_text', '')!r}, "
                f"page_number={c.get('page_number', '')})\n"
                f"{c.get('chunk_text', '')}"
            )
        joined = "\n\n".join(chunk_blocks)

        doc_summary_section = ""
        if doc_summary:
            doc_summary_section = f"\nDocument-level summary for context:\n{doc_summary}\n"

        custom_section = ""
        if custom_instructions:
            custom_section = (
                "\nAdditional guidance for this document set:\n"
                "IMPORTANT: You are running the CHUNK-METADATA stage. The chunk text is "
                "provided below — you MUST NOT alter, rewrite, or regenerate it. Only produce "
                "metadata about each chunk. Follow per-chunk classification / tagging guidance "
                "(clause_category, customer_specific_tags, etc.) from the prose below; ignore "
                "docmeta or document-level instructions.\n\n"
                + custom_instructions
                + "\n"
            )

        return f"""
You are a document analyst. Below are {len(chunks)} pre-assembled chunks from a document. The chunk text is FIXED — do not rewrite it.

Your task: for each chunk, emit a metadata object. Return a JSON array with EXACTLY {len(chunks)} objects in the same order as the chunks below.

Per-chunk fields (all required unless marked optional):
- "chunk_index": integer, 1-based, matching the CHUNK N label.
- "topic": a concise title (3-10 words) describing what this chunk covers.
- "summary": 1-2 sentences summarizing the chunk's content.
- "sub_topics": comma-separated list of sub-topics, or "" if none.
- "section_reference": exact legal/numerical citation (e.g. "Section 3(c)(ii)", "Article 12.4", "Exhibit G"), or "" if absent.
- "content_type": EXACTLY one of: "metrics" (quantitative data/tables/numbers), "narrative" (explanatory prose), "instructions" (step-by-step procedure), "comparisons" (parallel structures comparing items), "legal" (disclaimers, clauses, formal legal language), "concept" (term definitions), "other".
- "potential_questions": JSON array of 3-5 natural-language questions this chunk directly and completely answers. Each question must be answerable ONLY from this chunk's text. Do NOT write broad questions like "What does this lease say?" — be specific to the chunk's facts.
- "customer_specific_tags": JSON array of strings following the guidance below, or [] if none.
- "universal": pipe-separated universal headings applicable to the chunk (document title, dates, confidentiality), or "" if none.

Additional per-chunk fields:
- If the "Additional guidance for this document set" section below declares extra per-chunk fields (e.g. via `"field_name":` syntax), include them as additional keys on each chunk's metadata object. Use lowercase snake_case for these keys.
{doc_summary_section}{custom_section}
CRITICAL:
- Do NOT return chunk_text. It is assembled from source and is not your responsibility.
- Do NOT invent facts. Every question, tag, and summary must be grounded in the chunk text you're given.
- Output ONLY the JSON array. No markdown fences, no prose before or after.

---
{joined}
""".strip()
