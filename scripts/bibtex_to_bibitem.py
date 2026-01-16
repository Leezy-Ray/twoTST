"""
Convert BibTeX format to LaTeX \bibitem format
Handles @article, @book, @inproceedings, @inbook entry types
"""
import re
import sys


def parse_bibtex_entry(entry_text):
    """Parse a single BibTeX entry into a dictionary"""
    entry = {}
    
    # Extract entry type and key
    match = re.match(r'@(\w+)\{(\w+),', entry_text)
    if not match:
        return None
    
    entry['type'] = match.group(1)
    entry['key'] = match.group(2)
    
    # Extract all fields - handle nested braces properly
    pos = 0
    while pos < len(entry_text):
        # Find field name
        field_match = re.search(r'(\w+)\s*=\s*\{', entry_text[pos:])
        if not field_match:
            break
        
        field_name = field_match.group(1)
        start_pos = pos + field_match.end()
        
        # Find matching closing brace (handle nested braces)
        brace_count = 1
        end_pos = start_pos
        while end_pos < len(entry_text) and brace_count > 0:
            if entry_text[end_pos] == '{':
                brace_count += 1
            elif entry_text[end_pos] == '}':
                brace_count -= 1
            end_pos += 1
        
        if brace_count == 0:
            value = entry_text[start_pos:end_pos-1]  # Exclude closing brace
            # Remove LaTeX commands that might cause issues
            value = value.replace('\\L', 'L').replace('ukasz', 'ukasz')
            # Remove outer braces if present (handle nested braces)
            while value.startswith('{') and value.endswith('}') and value.count('{') == value.count('}'):
                value = value[1:-1]
            entry[field_name.lower()] = value.strip()
            pos = end_pos
        else:
            break
    
    return entry


def format_author_list(author_str):
    """Format author list: 'Last, First and Last, First' -> 'First Last; First Last'"""
    if not author_str:
        return ""
    
    # Handle special cases like {American Psychiatric Association} or {{American Psychiatric Association}}
    # Remove all outer braces
    author_str = author_str.strip()
    while author_str.startswith('{') and author_str.endswith('}') and author_str.count('{') == author_str.count('}'):
        author_str = author_str[1:-1].strip()
    
    # If it's an organization name (no comma, no 'and'), return as is
    if not ',' in author_str and not ' and ' in author_str:
        return author_str
    
    authors = []
    # Split by ' and ' or ' and'
    parts = re.split(r'\s+and\s+', author_str)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Check if it's "Last, First" format
        if ',' in part:
            last, first = part.split(',', 1)
            authors.append(f"{first.strip()} {last.strip()}")
        else:
            # Already in "First Last" format
            authors.append(part)
    
    # Replace "others" or "et al" with "et al."
    if len(authors) > 0:
        last_author = authors[-1].lower()
        if 'others' in last_author or 'et al' in last_author:
            authors[-1] = "et al."
    
    return '; '.join(authors)


def format_article(entry):
    """Format @article entry - MDPI style"""
    author = format_author_list(entry.get('author', ''))
    title = entry.get('title', '')
    journal = entry.get('journal', '')
    year = entry.get('year', '')
    volume = entry.get('volume', '')
    number = entry.get('number', '')
    pages = entry.get('pages', '')
    
    # Build the citation
    parts = []
    if author:
        parts.append(f"{author}.")
    if title:
        parts.append(f"{title}.")
    if journal:
        parts.append(f"\\textit{{{journal}}}")
    if year:
        parts.append(f"\\textbf{{{year}}}")
    if volume:
        if number:
            parts.append(f"\\textit{{{volume}}}, \\textit{{{number}}}")
        else:
            parts.append(f"\\textit{{{volume}}}")
    elif number:
        parts.append(f"\\textit{{{number}}}")
    if pages:
        parts.append(f"{pages}.")
    else:
        parts.append(".")
    
    return ' '.join(parts)


def format_book(entry):
    """Format @book entry"""
    author = format_author_list(entry.get('author', ''))
    title = entry.get('title', '')
    publisher = entry.get('publisher', '')
    year = entry.get('year', '')
    volume = entry.get('volume', '')
    pages = entry.get('pages', '')
    address = entry.get('address', '')
    
    parts = []
    if author:
        parts.append(f"{author}.")
    if title:
        parts.append(f"\\textit{{{title}}}")
    if publisher:
        if address:
            parts.append(f"; {publisher}: {address}")
        else:
            parts.append(f"; {publisher}")
    if year:
        parts.append(f", {year}")
    if volume:
        parts.append(f"; Volume {volume}")
    if pages:
        parts.append(f", pp. {pages}.")
    else:
        parts.append(".")
    
    return ' '.join(parts)


def format_inproceedings(entry):
    """Format @inproceedings entry"""
    author = format_author_list(entry.get('author', ''))
    title = entry.get('title', '')
    booktitle = entry.get('booktitle', '')
    year = entry.get('year', '')
    pages = entry.get('pages', '')
    address = entry.get('address', '')
    publisher = entry.get('publisher', '')
    volume = entry.get('volume', '')
    
    parts = []
    if author:
        parts.append(f"{author}.")
    if title:
        parts.append(f"{title}.")
    
    # Format as "In Proceedings of..."
    if booktitle:
        if 'Proceedings' in booktitle or 'Conference' in booktitle or 'Workshop' in booktitle:
            parts.append(f"In {booktitle}")
        else:
            parts.append(f"In \\textit{{{booktitle}}}")
    
    if address:
        parts.append(f", {address}")
    if year:
        parts.append(f", {year}")
    if pages:
        if pages.strip():
            parts.append(f"; pp. {pages}.")
        else:
            parts.append(".")
    elif volume:
        parts.append(f"; Volume {volume}.")
    else:
        parts.append(".")
    
    return ' '.join(parts)


def format_inbook(entry):
    """Format @inbook entry"""
    author = format_author_list(entry.get('author', ''))
    title = entry.get('title', '')
    booktitle = entry.get('booktitle', '')
    year = entry.get('year', '')
    pages = entry.get('pages', '')
    publisher = entry.get('publisher', '')
    address = entry.get('address', '')
    
    parts = []
    if author:
        parts.append(f"{author}.")
    if title:
        parts.append(f"{title}.")
    if booktitle:
        parts.append(f"In \\textit{{{booktitle}}}")
    if publisher:
        if address:
            parts.append(f"; {publisher}: {address}")
        else:
            parts.append(f"; {publisher}")
    if year:
        parts.append(f", {year}")
    if pages:
        parts.append(f"; pp. {pages}.")
    else:
        parts.append(".")
    
    return ' '.join(parts)


def convert_bibtex_to_bibitem(bibtex_file, output_file=None):
    """Convert BibTeX file to \bibitem format"""
    
    with open(bibtex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into entries
    # Pattern: @type{key, ... }
    entries = []
    current_entry = None
    brace_count = 0
    
    for line in content.split('\n'):
        # Check for new entry
        match = re.match(r'@(\w+)\{(\w+),', line)
        if match:
            if current_entry:
                entries.append(current_entry)
            current_entry = {'lines': [line], 'type': match.group(1), 'key': match.group(2)}
            brace_count = line.count('{') - line.count('}')
        elif current_entry:
            current_entry['lines'].append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and '}' in line:
                entries.append(current_entry)
                current_entry = None
    
    if current_entry:
        entries.append(current_entry)
    
    # Parse and format entries
    bibitems = []
    for entry_data in entries:
        entry_text = '\n'.join(entry_data['lines'])
        entry = parse_bibtex_entry(entry_text)
        
        if not entry:
            continue
        
        key = entry['key']
        entry_type = entry['type']
        
        # Format based on type
        if entry_type == 'article':
            formatted = format_article(entry)
        elif entry_type == 'book':
            formatted = format_book(entry)
        elif entry_type == 'inproceedings':
            formatted = format_inproceedings(entry)
        elif entry_type == 'inbook':
            formatted = format_inbook(entry)
        else:
            # Default formatting
            formatted = format_article(entry) if 'journal' in entry else format_book(entry)
        
        bibitems.append((key, formatted))
    
    # Generate output
    output_lines = [
        "\\reftitle{References}",
        "",
        "\\begin{thebibliography}{99}",
        ""
    ]
    
    for key, formatted in bibitems:
        output_lines.append(f"\\bibitem{{{key}}}")
        output_lines.append(f"{formatted}")
        output_lines.append("")
    
    output_lines.append("\\end{thebibliography}")
    
    output_text = '\n'.join(output_lines)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Converted {len(bibitems)} entries. Output saved to {output_file}")
    else:
        print(output_text)
    
    return output_text


def main():
    if len(sys.argv) < 2:
        print("Usage: python bibtex_to_bibitem.py <input.bib> [output.tex]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_bibtex_to_bibitem(input_file, output_file)


if __name__ == '__main__':
    main()
