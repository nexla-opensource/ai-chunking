import json
import os

# This is a simplified version of the sample JSON provided in your prompt
sample_json = {
  "children": [
    {
      "id": "/page/0/Page/561",
      "block_type": "Page",
      "html": "Page content reference",
      "children": [
        {
          "id": "/page/0/SectionHeader/0",
          "block_type": "SectionHeader",
          "html": "<h1>oaktree power opportunities fund vi, l.p.</h1>",
          "section_hierarchy": {
            "1": "/page/0/SectionHeader/0"
          },
          "images": {}
        },
        {
          "id": "/page/0/Picture/1",
          "block_type": "Picture",
          "html": "<p role=\"img\">Image description: The image shows the logo for Oaktree.</p>",
          "section_hierarchy": {
            "1": "/page/0/SectionHeader/0"
          },
          "images": {}
        },
        {
          "id": "/page/0/SectionHeader/2",
          "block_type": "SectionHeader",
          "html": "<h4>Summary of Fund Terms</h4>",
          "section_hierarchy": {
            "1": "/page/0/SectionHeader/0",
            "4": "/page/0/SectionHeader/2"
          },
          "images": {}
        },
        {
          "id": "/page/0/Text/3",
          "block_type": "Text",
          "html": "<p block-type=\"Text\">Reflects the Amended and Restated LPA dated November 16, 2021 <i>Confidential</i></p>",
          "section_hierarchy": {
            "1": "/page/0/SectionHeader/0",
            "4": "/page/0/SectionHeader/2"
          },
          "images": {}
        },
        {
          "id": "/page/0/SectionHeader/13",
          "block_type": "SectionHeader",
          "html": "<h1>investment objective</h1>",
          "section_hierarchy": {
            "1": "/page/0/SectionHeader/13"
          },
          "images": {}
        },
        {
          "id": "/page/0/Text/14",
          "block_type": "Text",
          "html": "<p block-type=\"Text\">A primary objective of the Fund is to realize substantial capital appreciation without subjecting principal to undue risk.</p>",
          "section_hierarchy": {
            "1": "/page/0/SectionHeader/13"
          },
          "images": {}
        },
        {
          "id": "/page/0/Text/15",
          "block_type": "Text",
          "html": "<p block-type=\"Text\">The Fund will seek to achieve this objective primarily by making investments in companies that focus primarily on providing products and services to owners of critical infrastructure, including electric power, natural gas, water, wastewater and other utility-related businesses.</p>",
          "section_hierarchy": {
            "1": "/page/0/SectionHeader/13"
          },
          "images": {}
        }
      ],
      "section_hierarchy": {
        "1": "/page/0/SectionHeader/13"
      },
      "images": null
    },
    {
      "id": "/page/1/Page/147",
      "block_type": "Page",
      "html": "Page content reference",
      "children": [
        {
          "id": "/page/1/SectionHeader/0",
          "block_type": "SectionHeader",
          "html": "<h2>fund financing</h2>",
          "section_hierarchy": {
            "1": "/page/0/SectionHeader/13",
            "2": "/page/1/SectionHeader/0"
          },
          "images": {}
        },
        {
          "id": "/page/1/Text/3",
          "block_type": "Text",
          "html": "<p block-type=\"Text\"><i>This report is being provided on a confidential basis solely for the information of those persons that have received this report in their capacity as limited partners or shareholders.</i></p>",
          "section_hierarchy": {
            "1": "/page/1/SectionHeader/2"
          },
          "images": {}
        },
        {
          "id": "/page/1/Text/4",
          "block_type": "Text",
          "html": "<p block-type=\"Text\"><i>U.S. securities laws (and the securities laws of certain non-U.S. jurisdictions) prohibit any person who has received material non-public information about a company from purchasing or selling securities of such company or from communicating such information to any other person.</i></p>",
          "section_hierarchy": {
            "1": "/page/1/SectionHeader/2"
          },
          "images": {}
        }
      ],
      "section_hierarchy": {
        "1": "/page/1/SectionHeader/2"
      },
      "images": null
    }
  ],
  "block_type": "Document"
}

def main():
    """Save the sample JSON to a file for testing."""
    output_file = "sample_pdf.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_json, f, indent=2, ensure_ascii=False)
    
    print(f"Sample JSON saved to {output_file}")
    print(f"You can test the chunker with: python chunker.py {output_file} output_chunks.json")

if __name__ == "__main__":
    main() 