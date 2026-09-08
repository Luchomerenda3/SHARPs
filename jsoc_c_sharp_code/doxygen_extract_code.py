# This extracts all the code from a set of Doxygen generated documentation
# where the code is embedded and highlighted. You really only need to use this
# when attempting to recover lost code and you still have the docs.

# Writes all code out into the original directory structure relative to where
# the script is executed.

# Run: `python extract_code_from_doxygen.py URL_TO_DOXYGEN_FILES_PAGE`
# e.g. `python extract_code_from_doxygen.py http://swf2svg.sourceforge.net/azar/doc/files.html`

import sys
import re
import os
from urllib.parse import urlparse

import requests
import lxml.html

listing = sys.argv[1]  # The files.html doxygen url
base_url = "/".join(listing.split("/")[:-1])

response = requests.get(listing)
root = lxml.html.fromstring(response.content)

file_nodes = root.xpath("//table/tr/td[1]/a[2]")
print(f"Found {len(file_nodes)} files to extract")

for node in file_nodes:
    code_url = base_url + "/" + node.attrib["href"]
    print(f"Fetching: {code_url}")
    response = requests.get(code_url)

    code_root = lxml.html.fromstring(response.content)
    
    # Extract filename from title tag instead of h1
    title_elements = code_root.xpath("//title")
    if not title_elements:
        print(f"Warning: Could not find title for {code_url}, skipping...")
        continue
    
    title_text = title_elements[0].text_content().strip()
    
    # Extract the file path from title - format is typically "Project: /path/to/file.cpp File Reference"
    # Try to extract the file path between colons or after the first colon
    match = re.search(r':\s*(.+?)\s+File Reference', title_text)
    if match:
        h1_text = match.group(1).strip()
    else:
        print(f"Warning: Could not parse filename from title '{title_text}', skipping...")
        continue

    # Find the link to the source code page - look for "Go to the source code of this file." link
    source_links = code_root.xpath("//a[contains(text(), 'Go to the source code of this file')]")
    if not source_links:
        print(f"Warning: Could not find 'Go to the source code of this file' link for {h1_text}, skipping...")
        continue
    source_href = source_links[0].attrib.get("href")
    if not source_href:
        print(f"Warning: Source link has no href for {h1_text}, skipping...")
        continue

    # Construct the full URL for the source page
    source_url = base_url + "/" + source_href
    print(f"Fetching source: {source_url}")
    source_response = requests.get(source_url)
    source_root = lxml.html.fromstring(source_response.content)

    base_path, filename = os.path.split(h1_text)

    # Extremely hacky way to make a windows/linux path relative
    base_path = re.sub("^([a-zA-Z]:)*/", "", base_path)
    try:
        os.makedirs(base_path)
    except FileExistsError as e:
        pass

    pre_elements = source_root.xpath("//pre")
    if pre_elements:
        # Old format with <pre> tag
        pre = pre_elements[0].text_content()
        code = re.sub("^[0-9]+ ", "", pre, flags=re.MULTILINE)
    else:
        # New Doxygen format with <div class="line"> elements
        line_elements = source_root.xpath("//div[@class='line']")
        if not line_elements:
            print(f"Warning: Could not find code in any format for {h1_text}, skipping...")
            continue

        code_lines = []
        for line_elem in line_elements:
            # Get text content and clean up HTML
            line_text = line_elem.text_content()
            # Remove line number prefix more aggressively
            # Match: optional whitespace, then digits only, then optional whitespace
            line_text = re.sub(r'^\s*\d+\s*', '', line_text)
            code_lines.append(line_text)
        
        code = '\n'.join(code_lines)

    print("Writing %s/%s" % (base_path, filename))
    with open("%s/%s" % (base_path, filename), "w") as f:
        f.write(code)