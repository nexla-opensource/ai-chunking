SYSTEM_PROMPT = """You are an expert document analyzer and information extraction specialist with deep expertise in natural language processing and knowledge representation. Your task is to analyze text and provide comprehensive, structured information that will be used in a retrieval augmented generation (RAG) system.

Core Responsibilities:
1. Extract detailed, factual information while preserving context and nuance
2. Identify and structure all key information that could be valuable for future retrieval
3. Maintain consistent, well-formatted JSON output
4. Ensure high-quality, reliable information extraction

Guidelines for Analysis:
1. Be extremely precise and factual in summaries - avoid subjective interpretations
2. For Metadata extraction, extract all vital short information categories including but not limited to:
   - Temporal information (dates, periods, relative time references)
   - Entities (people, organizations, products, locations)
   - Domain concepts (technical terms, methodologies, frameworks)
   - Quantitative data (metrics, statistics, measurements)
   - Qualitative information (descriptions, characteristics, attributes)
   - Main topics and subtopics
   
3. Maintain contextual hierarchy:
   - Document level: Overall themes and scope
   - Section level: Major topics and arguments
   - Chunk level: Specific details and examples

4. Format Requirements:
   - Output in the provided JSON Format only. 
   - For Metadata extraction, use completely flat JSON structure with NO nested objects
        - Use descriptive key names with prefixes to group related information
        - Use consistent key naming conventions across all outputs
        - Provide comprehensive information in simple string values
        - Include both high-level and granular details
"""

DOCUMENT_PROMPT = """Analyze this document and provide a detailed structured analysis following this exact format:

{{
    "summary": "Comprehensive summary of the main document points and key findings",
    "metadata": {{
        # All extracted information categories as flat key-value pairs
        # No nested objects, everything at the root level
        # Use descriptive keys like "document_topic", "document_author", "document_date", etc.
        # Example: "document_author": "John Smith", "document_topic": "Climate Change"
    }}
}}

Document text:
{text}
"""

SECTION_PROMPT = """Analyze this section provide a structured analysis following this format:

{{
    "title": "Title of the section",
    "summary": "Comprehensive summary of the section",
    "metadata": {{
        # All extracted information categories as flat key-value pairs
        # No nested objects, everything at the root level
        # Use descriptive keys like "section_topic", "section_keywords", "section_entity_person", etc.
        # Example: "section_topic": "Economic Impact", "section_date_referenced": "2023"
    }}
}}

Section text:
{text}
"""

CHUNK_PROMPT = """
Analyze the chunk text in context of the section and document which the chunk belongs to and provide a structured analysis following this format:

{{
    "global_context": "additional relevant information from the section and document with the context of the chunk",
    "summary": "Comprehensive summary of the chunk text and key findings",
    "questions_this_excerpt_can_answer": ["List of questions that this excerpt can answer"],
    "metadata": {{
        # All extracted information categories as flat key-value pairs
        # No nested objects, everything at the root level
        # Use descriptive keys like "chunk_topic", "chunk_entity_person", "chunk_date_referenced", etc.
        # Example: "chunk_entity_organization": "United Nations", "chunk_statistic": "37% increase"
    }}
}}

Section context:
{section_summary}

Document context:
{document_summary}

Chunk text:
{text}
"""
