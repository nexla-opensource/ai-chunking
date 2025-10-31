"""Prompts specific to Markdown AI Chunker."""


def footer_detection_prompt(chunk_text_with_line_numbers: str) -> str:
    """Generate prompt for detecting footer content in the last chunk.
    Uses semantic boundary detection without relying on specific keywords.
    """
    return f"""Detect the semantic boundary where footer content begins.

**Task:** Find where the document shifts from topical content to platform boilerplate.

**Semantic boundary indicators:**

Before the boundary (TOPICAL CONTENT):
- Focused on a specific subject/topic
- Provides information, analysis, or instruction
- Sections are interconnected and build on each other
- Advances reader's understanding of the topic

After the boundary (PLATFORM BOILERPLATE):
- Serves website functions (not the document's topic)
- Facilitates user actions or navigation
- Multiple unrelated sections grouped together
- Focus shifts from "learning" to "doing something on the platform"

**Detection strategy:**

1. Scan the document flow
2. Notice where topical coherence BREAKS
3. Identify where platform-serving sections CLUSTER
4. Mark the FIRST line of this cluster

**Key question:**
"Does this section advance the document's main argument/topic, or does it serve the website?"

**Common boundary patterns:**
- Article/tutorial content → Social sharing options
- Technical explanation → Newsletter subscription
- Topic discussion → Author biography
- Content conclusion → Site navigation menus
- Subject analysis → Related posts/articles

**Important:**
- Appendices that extend the topic = NOT footer
- Conclusions about the topic = NOT footer
- References/citations = NOT footer
- Platform engagement sections = FOOTER

**Return:**
{{"footer_start_line": N}} for first footer line, or {{"footer_start_line": null}} if none.

**Document text:**
{chunk_text_with_line_numbers}

Output JSON only."""

