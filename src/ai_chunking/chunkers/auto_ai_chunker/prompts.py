"""Module containing all prompt templates used in the AI chunking process."""

from typing import Dict, List, Any
import asyncio
from functools import lru_cache

@lru_cache(maxsize=1000)
def _format_headings_list(headings: tuple) -> str:
    """Format headings list for prompts. Uses tuple for hashable input."""
    return "\n".join(str(h) for h in headings)

@lru_cache(maxsize=1000)
def _format_text_lines(text_lines: tuple) -> str:
    """Format text lines for prompts. Uses tuple for hashable input."""
    return "\n".join(text_lines)

def bridge_headings_prompt(
    current_page_last_heading: str,
    current_page_last_heading_text: Dict,
    next_page_first_heading_text: Dict,
    current_page_headings: List[Dict],
    universal_headings: List[str],
) -> str:
    """Generate prompt for determining if headings bridge across pages."""
    # Convert mutable inputs to immutable for caching
    current_headings_tuple = tuple(str(h) for h in current_page_headings)
    universal_headings_tuple = tuple(universal_headings)
    
    # Format text blocks using cached functions
    current_text = _format_text_lines(tuple(f"{k}: {v}" for k, v in current_page_last_heading_text.items()))
    next_text = _format_text_lines(tuple(f"{k}: {v}" for k, v in next_page_first_heading_text.items()))
    headings_text = _format_headings_list(current_headings_tuple)

    return f"""
You have information about two consecutive parts of a document:

**Current Page's Final Heading**
- Heading Title: "{current_page_last_heading}"
- Text Under That Heading (line_number -> text):
  {current_text}

**Next Page Lines** (before any new heading appears):
{next_text}

**Universal Headings** (applies across the entire document):
{universal_headings_tuple}

**Current Page All Headings**:
{headings_text}

### Task

Decide if **any** lines at the start of the next page (before its first explicit heading) logically continue the final heading's topic from the current page. Additionally, if the lines do not strictly continue that final heading but still partially or indirectly relate to **other** headings on the current page, classify them as `"relevant"` to those headings.

Thus, your output can be:
- **"bridging": "true"**  
  if at least some lines obviously continue the final heading's content (partial or total continuation).
- **"bridging": "false"**  
  if the next page text is a new topic or is fully explained by universal headings, disclaimers, or otherwise not referencing the old heading(s).
- **"bridging": "relevant"**  
  if the text is not continuing the final heading but still references or aligns with **other** headings from the current page. In such a case, you also specify which headings it relates to.

Return output in JSON format:
{{
  "bridging": "true" | "false" | "relevant",
  "start_index": <integer or null>,
  "rationale": "<string>",
  "most_relevant_headings": ["heading title 1", ...]
}}
"""

@lru_cache(maxsize=1000)
def semantic_headings_prompt(
    text_with_line_numbers: str, 
    page_number: str, 
    total_number_of_lines: int
) -> str:
    """Generate prompt for extracting semantic headings from text."""
    return f"""
You are given a single page (page number = {page_number}) from a document, with each line numbered from 1 to {total_number_of_lines}. Your task is to:

1. Identify **all headings** on this page.
2. Classify each heading into one of the following categories:
   - **global_heading**: A heading applying to the entire document (commonly near the top of the *first* page; e.g., a main title, date setting global context).
   - **page_heading**: A heading that specifically introduces the content of this particular page (e.g., a large or uppercase title at the top of the page but not global to the entire file).
   - **section_heading**: A heading introducing a major section on this page (e.g., "DISCLOSURES," "NOTES," "PERFORMANCE METRICS"). 
   - **footer_heading**: A heading or label found at the bottom of the page that serves a structural or reference purpose (often repeated disclaimers, page labels). 

### Guidelines:

**A. Handling Multiple Headings on the Same Page**
- There might be several headings scattered throughout the page. **List every heading** you detect, even if they appear close together.

**B. Determining Heading Types**
#### 1. Global Heading
- **Definition**: A heading that applies to the entire document. Commonly near the top of the *first* page (page_number == 1), but it can also appear on other pages if it explicitly references universal context (e.g. "2024 Annual Report").
- **Identifying Cues**:
  - Large or bold line at the **very top** (especially if `page_number == 1`).  
  - The heading might declare a **core identity** of the file: a main title, fund name, or overarching theme.  
  - Could be repeated on every page if it's a strong branding element (though this is less common).
- **Classification**:
  - If you see a heading that strongly suggests a universal scope for the entire file (like "Filed Date 2024-09-17"), label it `global_heading`.

#### 2. Page Heading
- **Definition**: A heading that specifically introduces the content of the current page only (not the entire document).
- **Identifying Cues**:
  - Typically near the **top** of the page (especially if not page 1).
  - Might say "SEPTEMBER 2024 FACTSHEET," or some top-of-page section name that is unique to this page's content.
  - Not obviously global (i.e., it references a topic for this page alone).
- **Classification**:
  - Label it `page_heading` if it sets the theme for *this page's* content and doesn't appear to apply to the entire file.

#### 3. Section Heading
- **Definition**: A heading that introduces a **major section or subsection** of content on this page.
- **Identifying Cues**:
  - Usually **uppercase** or bold, often followed by relevant data, paragraphs, bullet points, or tables.
  - Examples: "PERFORMANCE DISCLOSURES", "METRICS OVERVIEW".
  - Appears *below* any global or page heading; it typically segments the page into distinct topics or chapters.
- **Classification**:
  - If the heading is not universal (global) nor specifically top-of-page (page heading), but introduces a chunk of content on the page, label it `section_heading`.

#### 4. Footer Heading
- **Definition**: A heading or label at the **bottom** of the page that serves a structural or reference purpose (e.g., disclaimers, page numbering).
- **Identifying Cues**:
  - Likely repeated or appears on multiple pages, or purely references a structural note (e.g., "Page 2 of 10," "Confidential – For Internal Use," "Document ID: 1234," etc.).
  - Not a typical content-bearing heading—more so a printed line at the page's foot.
- **Classification**:
  - Label it `footer_heading` if it's near the bottom lines and primarily serves as a structural or repeated note rather than introducing new content.

**C. Formatting and Position Cues**
- **Uppercase/bold** on a standalone line often signals a heading.  
- **Location**: 
  - A large heading near the top might be `global_heading`, `page_heading`, or **bridge_heading** (if it explicitly indicates continuation).
  - A heading near the bottom could be `footer_heading` or **bridge_heading** (if referencing next-page continuation).
- **Textual Clues**:  
  - Words like "(continued)," "from previous page," or "see next page" strongly suggest a bridge heading.
  - "Draft," "Page X of Y," or disclaimers at the bottom often suggest a `footer_heading`.

**D. Line Number Validity**
- Your `start_index` must be **within the range** 1..{total_number_of_lines}.  
- If you detect multiple headings on the same line, they should each have the same `start_index`.  

**E. Overlapping or Repeated Headings**
- If the same heading text appears more than once on different lines, treat each occurrence as a separate heading with its own line numbers.
- If a heading repeats at both the top and bottom of the page, classify accordingly (e.g., one might be `page_heading`, the other a `footer_heading`).

### Output Format:
Return a JSON object:

{{
  "headings": [
    {{
      "title": "Heading Title",
      "start_index": <line_number>,
      "heading_type": "global_heading" | "page_heading" | "section_heading" | "footer_heading",
      "page_number": <page_number>,
    }}
  ]
}}

Document: {text_with_line_numbers}
Page Number: {page_number}
Total number of lines: {total_number_of_lines}
"""

@lru_cache(maxsize=1000)
def global_headings_prompt(
    global_headings: tuple,  # Tuple of heading objects
    footer_headings: tuple,  # Tuple of heading objects
    source: str
) -> str:
    """Generate prompt for determining universal headings.
    
    Args:
        global_headings: Tuple of heading objects from across the document
        footer_headings: Tuple of heading objects from page footers
        source: Source document identifier
        
    Returns:
        Formatted prompt string
    """
    # Convert tuples to string representations for the prompt
    global_headings_str = "\n".join(str(h) for h in global_headings)
    footer_headings_str = "\n".join(str(h) for h in footer_headings)

    return f"""
You are provided with two types of headings from a multi-page document:

1. **Global Headings:** Headings marked as applying to the entire document (e.g., main dates, universal disclaimers or titles).  
2. **Footer Headings:** Headings or labels at the bottom of each page, possibly repeated disclaimers or structural notes.

Additionally you are also provided with the source info of the document.

Your task is to produce a **unique list of headings** that should be included in the metadata of each content chunk for the entire document. In other words:
- De-duplicate any repeated headings that appear on multiple pages.
- Include only those headings that apply universally.
- Exclude any headings that are page-specific or irrelevant to the entire document.
- You may use **source info** to help unify or refine headings if it provides clues about the official or complete name.

### Guidelines

1. **Input:**  
   A JSON object with two arrays:  
   - `"global_headings"`: An array of objects each having `title` and `start_index` representing all global headings found across the file.  
   - `"footer_headings"`: An array of objects, each having `title` and `start_index representing all footer headings found across the file.
   - `"source_info"`: A string representing the source of the document.

2. **Deduplication:**  
   - If the same global heading or footer heading appears on multiple pages, list it only once in the final output.
   - When in doubt, compare headings in a case-insensitive manner, trimming whitespace or punctuation if needed.
   - Multiple headings could represent the same concept for Retrieval. For example United States and United States of America are same and we should only keep one of them.

3. **Filtering by Relevance:**  
   - For **footer_headings**, determine if the footer heading can be universal or page-specific. 
   - Typically footer_headings that are Notes or Instructions are page-specific.

4. **Use Source Info:**  
   - Extract key headings from the source info if it would add value to the final list.

5. **Output Format:**  
   Return a JSON object like:
   {{
     "universal_headings": [
       {{ "title": "Unique Global Heading 1", "page_number": 1 }},
       {{ "title": "Unique Global Heading 2", "page_number": 2 }},
       {{ "title": "Unique Global Footer Heading 1", "page_number": 3 }}
     ]
   }}

   Global Headings: {global_headings_str}
   Footer Headings: {footer_headings_str}
   Source Info: {source}
"""

@lru_cache(maxsize=1000)
def semantic_grouping_prompt(
    text_with_line_numbers: str,
    headings: tuple,  # Convert list to tuple for hashable input
    total_number_of_lines: int
) -> str:
    """Generate prompt for semantic grouping of text."""
    return f"""
Read the document below and extract a `StructuredDocument` object from it. The goal is to group lines into meaningful sections and sub-sections based on their topics, using detected headings to guide the structure. This grouping will be used for question-answering purposes, so ensure that sections and sub-sections are comprehensive, logically grouped, and capture all relevant information.

In addition, classify each section and sub-section by assigning a `content_type` from the following set:

- **Metrics**  
  Content primarily focused on quantitative data such as performance metrics, numerical results, percentages, returns, or other factual statistics.

- **Narrative**  
  Explanatory, descriptive, or analytical text, including background information, commentary, reasoning, and general discussion not limited to metrics.

- **Instructions**  
  Steps, procedures, guidelines, or recommendations on how to perform tasks or interpret data.

- **Comparisons**  
  Content that explicitly compares multiple datasets, entities, or periods, highlighting similarities, differences, or changes over time.

- **Legal Disclaimers**  
  Regulatory notices, compliance statements, warnings, or other legally required information that sets conditions or context for the usage of the provided information.

- **Concept Definition**  
  Sections that define key terms, concepts, frameworks, or theoretical underpinnings.

---

### **Heading Detection & Usage**

Use headings to decide section boundaries. Consider these heading types (you may infer them from formatting cues or from prior heading detection):

1. **global_heading**  
   - Typically at or near the top of the first page.  
   - Sets the context for the entire document (e.g., a fund name, a main title, a date).  
   - If present, everything else might fall under this global context.

2. **page_heading**  
   - A heading found near the top of a page (not necessarily the first).  
   - Applies to the content of the current page but may not represent the entire file.

3. **section_heading**  
   - Introduces a major content block within a page (e.g., "PERFORMANCE DISCLOSURES," "ENDNOTES").  
   - Helps partition the page into logical topics.

4. **footer_heading**  
   - A heading or short label at the bottom of the page (e.g., a repeated disclaimer or page label).  
   - Usually not a main content topic.

5. **bridge_heading**  
   - A heading that appears on the current page and is a continuation of the previous page's heading.
   - Use this as the section title when it is present.

When you identify a heading line:
- **Check** how it relates to previous headings (is it a sub-topic of an already detected heading, or does it start a new main topic?).
- **Decide** whether it signals the start of a new section or a sub-section.

---

### Guidelines:
**Sections:**
- Each section should focus on a single main concept or topic.
- If a heading suggests a top-level topic (e.g., a global or page-level heading), treat that as the parent section.
- Sections must ensure that all content relevant to a topic is grouped together, even if spread across multiple lines or formats.
- If a topic is carried over to the next page, include a `continued: true` tag in the JSON for that section.
- Assign a `content_type` tag to each section that best describes its primary semantic nature.

**Sub-sections:**
- If a section contains hierarchical sub-headings or structured content (e.g., indented bullets or sub-items), treat them as sub-sections.
- Sub-sections must belong to a parent section and must have descriptive titles that capture their specific focus.
- Sub-sections should ensure that questions about specific details are answerable without requiring information from unrelated sections.
- For topics continued across pages, add the `continued: true` tag to the corresponding sub-section in the JSON structure.
- Assign a `content_type` tag to each sub-section, following the same classification rules.

**Use of Formatting Cues:**
- **Uppercase or bold headings**: Consider them as potential top-level sections.
- **Numbers and bullet points following a heading**: Suggest that the heading is a main section, and the following items might be part of that section's content or sub-sections.
- **A line that stands out in formatting placed before lists or explanatory text**: Treat this as a main heading.

**Use of Line Numbers:**
- Each section and sub-section must have a `start_index` to indicate the line where it begins (inclusive). `start_index` should start from 1 and not 0.
- The first section must begin at the first line of the document.
- Sections and sub-sections must together cover the entire document, leaving no gaps.
- `start_index` and `end_index` must be within the range of `total_number_of_lines`.

**Descriptive Titles:**
- Titles for sections and sub-sections must succinctly summarize their content.
- If a heading at the start of the document represents the overarching topic for all subsequent sections, reflect that in the parent section's title or in sub-sections. If all following content falls under this initial heading, treat that heading as the parent section title and consider merging smaller, related sub-sections under it.
- Ensure titles are detailed enough to help identify the content's focus at a glance.
  - For a section listing performance metrics, use "Performance Metrics Overview" and `content_type: "Metrics"`.
  - For a sub-section defining a key concept, use "Definition of Beta" and `content_type: "Concept Definition"`.

**Summary:**
- Each section and sub-section must have a `summary` field that is a concise summary of the section or sub-section. Keep it short and to the point.

**Completeness for Question-Answering:**
- Group related content together for comprehensiveness.
- Avoid splitting logically related content across multiple sections unless it creates meaningful sub-sections of a larger topic.

### Output Format:
Return a JSON object in the following structure:
{{
  "sections": [
    {{
      "title": "Main Section Title",
      "start_index": <line_number>,
      "continued": <true_or_false>,
      "content_type": "Metrics" | "Narrative" | "Instructions" | "Comparisons" | "Legal Disclaimers" | "Concept Definition",
      "summary": "Concise Summary of the section",
      "sub_sections": [
        {{
          "title": "Sub-section Title",
          "start_index": <line_number>,
          "continued": <true_or_false>,
          "content_type": "Metrics" | "Narrative" | "Instructions" | "Comparisons" | "Legal Disclaimers" | "Concept Definition",
          "summary": "Concise Summary of the sub-section",
        }}
      ]
    }}
  ]
}}

### Additional Notes:
- Sections and sub-sections can vary in length but must logically group related content.
- If the document contains a mix of text types (e.g., some narrative, some data), choose the content_type that dominates that section or sub-section.
- Ensure hierarchical relationships (e.g., bullets under a main topic) are captured accurately to maintain logical organization.
- Use the continued tag to indicate if a topic is carried over to the next page.
- For tricky cases (like a note block between headings), rely on formatting signals (uppercase headings, standalone lines, followed by details or bullet points) to determine the correct hierarchy.

Document: {text_with_line_numbers}
Headings: {headings}
Total number of lines: {total_number_of_lines}
"""

@lru_cache(maxsize=1000)
def enrich_prompt(
    page_number: int,
    page_text: str,
    table_data: tuple  # Convert list to tuple for hashable input
) -> str:
    """Generate prompt for enriching page content with table data."""
    # Convert table data to string format for caching
    table_info = "\n".join(
        f"Table {i+1}:\n" + "\n".join(
            f"{k}: {v}" for k, v in row
        )
        for i, table in enumerate(table_data)
        for row in table
    )

    return f"""
    You are an assistant tasked with enriching page content based on table data.
    
    Page Number: {page_number}
    Page Content: {page_text}
    Table Data: {table_info}

    Output Requirements:
    - Enrich the Page Content using Table Data.
    - Accurately establish relationships between the table entries and the text content to make the information complete, detailed, and coherent.
    - Ensure no information is skipped or removed from the original Page Content unless explicitly required for clarity.
    - Maintain logical flow and readability in the enriched content.
    -  The enriched text must include both narrative improvements and statistical insights where applicable.

    Instructions:
    - Retain all key information from the original Page Content.
    - Identify where the Table Data provides additional context or supports claims in the text and seamlessly integrate it.
    - When enriching the content, ensure clarity and provide additional context as needed without introducing inconsistencies or irrelevant details.
    - Return the final enriched content directly, with no comments or explanations.

    Output Format:
    - Return the output in JSON format with the following keys:
        {{
            "enriched_text": The final enriched text.
            "rationale": Rationale for the enriched text.
        }}
    """ 