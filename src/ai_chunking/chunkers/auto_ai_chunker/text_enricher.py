import logging
from typing import Dict, List, Optional
import asyncio

from ai_chunking.llm.base import StructuredLLMClient
from .models.document import EnrichedPageData, Heading, Page, TableData, HeadingType, HeadingLevel
from .models.llm_responses import HeadingAnalysis, TableAnalysis, EnrichedTextResponse
from .prompts import enrich_prompt, semantic_headings_prompt

logger = logging.getLogger(__name__)


class TextEnricher:
    """Enriches document text with tables and contextual information."""

    def __init__(self, llm_client: StructuredLLMClient):
        """Initialize text enricher.
        
        Args:
            llm_client: LLM client for text analysis (should be small/fast model)
        """
        self.llm_client = llm_client
        self._cache = {}
        logger.debug("TextEnricher initialized with small/fast LLM model")

    def _clear_cache(self):
        """Clear internal caches to free memory."""
        self._cache.clear()
        logger.debug("TextEnricher cache cleared")

    async def enrich_page(
        self,
        page: Page,
        table_data: Optional[List[TableData]] = None
    ) -> EnrichedPageData:
        """Enrich a page with tables and contextual information.
        
        Args:
            page: Page to enrich
            table_data: Optional table data to incorporate
            
        Returns:
            Enriched page data
        """
        try:
            logger.info(f"Starting enrichment for page {page.page_number}")
            logger.debug(f"Page {page.page_number} initial text length: {len(page.text)}")
            
            # Format table data for prompt
            formatted_tables = []
            if table_data:
                logger.debug(f"Found {len(table_data)} tables for processing")
                for table in table_data:
                    # Convert each row to a tuple of sorted items for hashing
                    formatted_rows = []
                    for row in table.rows:
                        # Sort items and convert to tuple for hashing
                        sorted_items = tuple(sorted((str(k), str(v)) for k, v in row.items()))
                        formatted_rows.append(sorted_items)
                    formatted_tables.append(tuple(formatted_rows))
            
            # Get enrichment prompt using the imported function
            prompt = enrich_prompt(
                page_number=page.page_number,
                page_text=page.text,
                table_data=tuple(formatted_tables)  # Convert to tuple for caching
            )
            
            # Get enriched text from LLM
            try:
                enriched_response = await self.llm_client.structured_generate(
                    prompt=prompt,
                    response_model=EnrichedTextResponse
                )
                enriched_text = enriched_response.enriched_text
                logger.debug(f"Successfully enriched text for page {page.page_number}")
            except Exception as e:
                logger.warning(f"Error enriching text for page {page.page_number}: {str(e)}")
                enriched_text = page.text
            
            # Extract and classify headings
            logger.debug(f"Extracting headings for page {page.page_number}")
            headings = await self._extract_headings(enriched_text, page.page_number)
            logger.info(f"Found {len(headings)} headings in page {page.page_number}")
            
            # Create enriched text lines map
            enriched_text_lines = enriched_text.split('\n')
            enriched_text_lines_map = {i+1: line for i, line in enumerate(enriched_text_lines)}
            
            enriched_page = EnrichedPageData(
                page_number=page.page_number,
                source=page.source or "",
                enriched_text=enriched_text,
                headings=headings,
                tables=table_data or [],
                enriched_text_lines_map=enriched_text_lines_map
            )
            logger.info(f"Successfully enriched page {page.page_number}")
            logger.debug(f"Final enriched text length: {len(enriched_text)}")
            return enriched_page
            
        except Exception as e:
            logger.error(f"Error enriching page {page.page_number}: {str(e)}")
            raise
        finally:
            self._clear_cache()

    async def _extract_headings(self, text: str, page_number: int) -> List[Heading]:
        """Extract and classify headings from text using LLM.
        
        Args:
            text: Text to analyze
            page_number: Page number being analyzed
            
        Returns:
            List of identified headings
        """
        lines = text.split("\n")
        enriched_text_with_line_numbers = ""
        for i, line in enumerate(lines):
            enriched_text_with_line_numbers += f"{i+1}. {line}\n"

        # Get semantic headings using LLM
        headings_prompt = semantic_headings_prompt(
            enriched_text_with_line_numbers,
            str(page_number),
            len(lines)
        )

        try:
            headings_response = await self.llm_client.structured_generate(
                prompt=headings_prompt,
                response_model=HeadingAnalysis
            )
            
            if not headings_response or not headings_response.headings:
                logger.warning(f"No headings found for page {page_number}")
                return []

            # Convert response headings to Heading objects
            headings = []
            for h in headings_response.headings:
                # Validate heading type
                heading_type = h.heading_type or HeadingType.SECTION
                if heading_type not in HeadingType.__members__.values():
                    logger.warning(f"Invalid heading type {heading_type}, defaulting to section_heading")
                    heading_type = HeadingType.SECTION

                # Create Heading object
                heading = Heading(
                    text=h.text,
                    level=h.level or HeadingLevel.H2,  # Default to H2 if not specified
                    line_number=h.line_number,
                    is_bridge=h.is_bridge or False,
                    heading_type=heading_type,
                    page_number=page_number
                )
                headings.append(heading)

            # Sort headings by line number
            headings.sort(key=lambda h: h.line_number)
            return headings

        except Exception as e:
            logger.error(f"Error extracting headings for page {page_number}: {str(e)}")
            return [] 